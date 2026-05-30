#include <cstdint>
#include <iostream>

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
