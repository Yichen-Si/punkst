#include <cstdint>
#include <iostream>

#include "image_pmtiles.hpp"
#include "image_utils.hpp"

namespace {

int image_output_disabled(const char* command) {
    std::cerr << command << " is unavailable because this build was configured "
              << "with ENABLE_IMAGE_OUTPUT=OFF. Rebuild with "
              << "-DENABLE_IMAGE_OUTPUT=ON to use this command.\n";
    return 1;
}

} // namespace

int32_t cmdDrawPixelFactors(int32_t, char**) {
    return image_output_disabled("draw-pixel-factors");
}

int32_t cmdDrawLowresFactors(int32_t, char**) {
    return image_output_disabled("draw-lowres-factors");
}

int32_t cmdDrawPixelFeatures(int32_t, char**) {
    return image_output_disabled("draw-pixel-features");
}

int32_t cmdImage2Pmtiles(int32_t, char**) {
    return image_output_disabled("image2pmtiles");
}

void save_png_rgb8(const std::string&, const Image2D<Rgb8>&) {
    image_output_disabled("PNG image output");
}

Image2D<Rgb8> load_png_rgb8(const std::string&) {
    image_output_disabled("PNG image input");
    return Image2D<Rgb8>();
}

Image2D<Rgba8> load_png_rgba8(const std::string&) {
    image_output_disabled("PNG image input");
    return Image2D<Rgba8>();
}

std::string encode_png_rgb8(const Image2D<Rgb8>&) {
    image_output_disabled("PNG image encoding");
    return std::string();
}

std::string encode_png_rgba8(const Image2D<Rgba8>&) {
    image_output_disabled("PNG image encoding");
    return std::string();
}

Image2D<Rgba8> decode_png_rgba8(const std::string&) {
    image_output_disabled("PNG image decoding");
    return Image2D<Rgba8>();
}

namespace image_pmtiles {

void validate_options(const Options&) {
    image_output_disabled("image2pmtiles");
}

Asset write_image_pmtiles(const Options&) {
    image_output_disabled("image2pmtiles");
    return Asset{};
}

} // namespace image_pmtiles
