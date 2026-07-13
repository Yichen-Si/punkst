#include "image_utils.hpp"

#include "utils.h"
#include "utils_sys.hpp"

#include <array>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <png.h>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <zlib.h>

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

uint32_t quantile_threshold_u32(std::vector<uint32_t>& values, double quantile) {
    if (values.empty()) {
        return 255;
    }
    const double q = std::clamp(quantile, 0.0, 1.0);
    size_t nth = 0;
    if (q >= 1.0) {
        nth = values.size() - 1;
    } else {
        nth = static_cast<size_t>(
            std::floor(q * static_cast<double>(values.size())));
        nth = std::min(nth, values.size() - 1);
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(nth),
        values.end());
    return std::max<uint32_t>(1, values[nth]);
}

uint8_t linear_adjusted_intensity_u8(uint32_t raw, uint32_t threshold) {
    if (raw == 0) {
        return 0;
    }
    const uint32_t safeThreshold = std::max<uint32_t>(1, threshold);
    if (raw > safeThreshold) {
        return 255;
    }
    const uint64_t scaled = static_cast<uint64_t>(raw) * 255u / safeThreshold;
    return static_cast<uint8_t>(std::min<uint64_t>(255u, scaled));
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
    Image2D<Rgba8> rgba = load_png_rgba8(filename);
    Image2D<Rgb8> image(rgba.height(), rgba.width());
    for (int y = 0; y < rgba.height(); ++y) {
        for (int x = 0; x < rgba.width(); ++x) {
            const Rgba8& p = rgba(y, x);
            image(y, x) = Rgb8{p.r, p.g, p.b};
        }
    }
    return image;
}

Image2D<Rgba8> load_png_rgba8(const std::string& filename) {
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
    if (!(color_type & PNG_COLOR_MASK_ALPHA)) {
        png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
    }

    png_read_update_info(png_ptr, info_ptr);
    Image2D<Rgba8> image(static_cast<int>(height), static_cast<int>(width));
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

struct PngStringReader {
    const std::string* data = nullptr;
    size_t offset = 0;
};

void png_string_read(png_structp png_ptr, png_bytep data, png_size_t length) {
    auto* in = static_cast<PngStringReader*>(png_get_io_ptr(png_ptr));
    if (!in || !in->data || length > in->data->size() - in->offset) {
        png_error(png_ptr, "truncated PNG string");
    }
    std::memcpy(data, in->data->data() + in->offset, length);
    in->offset += length;
}

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

std::string encode_png_rgba8(const Image2D<Rgba8>& image) {
    static_assert(sizeof(Rgba8) == 4, "Rgba8 must be tightly packed");
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
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    std::vector<png_bytep> rows(static_cast<size_t>(image.height()));
    const auto& pixels = image.data();
    for (int y = 0; y < image.height(); ++y) {
        rows[static_cast<size_t>(y)] = reinterpret_cast<png_bytep>(
            const_cast<Rgba8*>(&pixels[static_cast<size_t>(y) * static_cast<size_t>(image.width())]));
    }
    png_write_image(png_ptr, rows.data());
    png_write_end(png_ptr, nullptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return out;
}

Image2D<Rgba8> decode_png_rgba8(const std::string& png) {
    if (png.size() < 8 || png_sig_cmp(reinterpret_cast<png_const_bytep>(png.data()), 0, 8) != 0) {
        error("Input string is not a PNG image");
    }
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        error("Error initializing PNG reader");
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        error("Error initializing PNG metadata");
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        error("Error decoding PNG image");
    }
    PngStringReader reader{&png, 0};
    png_set_read_fn(png_ptr, &reader, png_string_read);
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
    if (!(color_type & PNG_COLOR_MASK_ALPHA)) {
        png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
    }

    png_read_update_info(png_ptr, info_ptr);
    Image2D<Rgba8> image(static_cast<int>(height), static_cast<int>(width));
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    auto& pixels = image.data();
    for (png_uint_32 y = 0; y < height; ++y) {
        rows[static_cast<size_t>(y)] = reinterpret_cast<png_bytep>(
            &pixels[static_cast<size_t>(y) * static_cast<size_t>(width)]);
    }
    png_read_image(png_ptr, rows.data());
    png_read_end(png_ptr, nullptr);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return image;
}

namespace {

struct TiffEntry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint64_t count = 0;
    uint64_t valueOffset = 0;
    std::array<uint8_t, 8> inlineBytes{};
    size_t inlineSize = 0;
};

size_t tiff_type_size(uint16_t type) {
    switch (type) {
    case 1:
    case 2:
    case 6:
    case 7:
        return 1;
    case 3:
    case 8:
        return 2;
    case 4:
    case 9:
    case 11:
    case 13:
        return 4;
    case 5:
    case 10:
    case 12:
    case 16:
    case 17:
    case 18:
        return 8;
    default:
        error("%s: unsupported TIFF field type %u", __func__, static_cast<unsigned>(type));
        return 0;
    }
}

uint64_t read_scalar_from_bytes(const uint8_t* data, uint16_t type, bool le) {
    switch (type) {
    case 1:
    case 2:
    case 6:
    case 7:
        return data[0];
    case 3:
    case 8:
        return read_u16_buf(data, le);
    case 4:
    case 9:
    case 11:
    case 13:
        return read_u32_buf(data, le);
    case 16:
    case 17:
    case 18:
    case 5:
    case 10:
    case 12:
        return read_u64_buf(data, le);
    default:
        error("%s: unsupported TIFF scalar type %u", __func__, static_cast<unsigned>(type));
        return 0;
    }
}

std::vector<uint64_t> read_values(RandomAccessFile& file, const TiffEntry& e, bool le) {
    const size_t step = tiff_type_size(e.type);
    if (e.count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / step)) {
        error("%s: TIFF tag %u has too many values", __func__, static_cast<unsigned>(e.tag));
    }
    const size_t valueBytes = static_cast<size_t>(e.count) * step;
    std::vector<uint8_t> buf;
    const uint8_t* data = nullptr;
    if (valueBytes <= e.inlineSize) {
        data = e.inlineBytes.data();
    } else {
        buf = file.read(e.valueOffset, valueBytes);
        data = buf.data();
    }
    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(e.count));
    for (uint64_t i = 0; i < e.count; ++i) {
        out.push_back(read_scalar_from_bytes(data + static_cast<size_t>(i) * step, e.type, le));
    }
    return out;
}

image_tiff::TiffLevel parse_tiff_level(RandomAccessFile& file, const image_tiff::TiffInfo& info,
    uint64_t ifdOffset, std::vector<uint64_t>* subIfds) {
    const size_t countSize = info.bigTiff ? 8 : 2;
    const size_t entrySize = info.bigTiff ? 20 : 12;
    const size_t inlineSize = info.bigTiff ? 8 : 4;
    if (ifdOffset >= file.size()) {
        error("%s: IFD offset outside TIFF file %s", __func__, file.path().string().c_str());
    }
    const uint64_t nEntries = info.bigTiff
        ? file.read_u64_at(ifdOffset, info.littleEndian)
        : file.read_u16_at(ifdOffset, info.littleEndian);
    if (nEntries > 1000000) {
        error("%s: suspicious TIFF IFD entry count %" PRIu64, __func__, nEntries);
    }
    const uint64_t entriesStart = ifdOffset + countSize;
    if (nEntries > (std::numeric_limits<uint64_t>::max() - entriesStart) / entrySize ||
            entriesStart + nEntries * entrySize > file.size()) {
        error("%s: truncated TIFF IFD in %s", __func__, file.path().string().c_str());
    }
    const std::vector<uint8_t> entriesBuf = file.read(entriesStart,
        static_cast<size_t>(nEntries) * entrySize);
    std::map<uint16_t, TiffEntry> entries;
    for (uint64_t i = 0; i < nEntries; ++i) {
        const size_t off = static_cast<size_t>(i) * entrySize;
        TiffEntry e;
        e.tag = read_u16(entriesBuf, off, info.littleEndian);
        e.type = read_u16(entriesBuf, off + 2, info.littleEndian);
        e.count = info.bigTiff
            ? read_u64(entriesBuf, off + 4, info.littleEndian)
            : read_u32(entriesBuf, off + 4, info.littleEndian);
        const size_t valueField = off + (info.bigTiff ? 12 : 8);
        e.inlineSize = inlineSize;
        std::copy(entriesBuf.begin() + static_cast<std::ptrdiff_t>(valueField),
            entriesBuf.begin() + static_cast<std::ptrdiff_t>(valueField + inlineSize),
            e.inlineBytes.begin());
        e.valueOffset = info.bigTiff
            ? read_u64_buf(e.inlineBytes.data(), info.littleEndian)
            : read_u32_buf(e.inlineBytes.data(), info.littleEndian);
        entries[e.tag] = e;
    }

    auto one = [&](uint16_t tag, uint64_t def = 0) -> uint64_t {
        auto it = entries.find(tag);
        if (it == entries.end()) return def;
        std::vector<uint64_t> vals = read_values(file, it->second, info.littleEndian);
        if (vals.empty()) {
            error("%s: TIFF tag %u has no values", __func__, static_cast<unsigned>(tag));
        }
        return vals.front();
    };
    auto many = [&](uint16_t tag) -> std::vector<uint64_t> {
        auto it = entries.find(tag);
        if (it == entries.end()) return {};
        return read_values(file, it->second, info.littleEndian);
    };
    auto many_u16 = [&](uint16_t tag) -> std::vector<uint16_t> {
        std::vector<uint16_t> out;
        for (uint64_t v : many(tag)) {
            out.push_back(static_cast<uint16_t>(v));
        }
        return out;
    };

    image_tiff::TiffLevel level;
    level.ifdOffset = ifdOffset;
    level.width = one(256);
    level.height = one(257);
    level.compression = static_cast<uint16_t>(one(259, 1));
    level.photometric = static_cast<uint16_t>(one(262, 0));
    level.samplesPerPixel = static_cast<uint16_t>(one(277, 1));
    level.planarConfig = static_cast<uint16_t>(one(284, 1));
    level.bitsPerSample = many_u16(258);
    if (level.bitsPerSample.empty()) {
        level.bitsPerSample.push_back(1);
    }
    level.sampleFormat = many_u16(339);
    if (level.sampleFormat.empty()) {
        level.sampleFormat.assign(level.bitsPerSample.size(), 1);
    }
    level.extraSamples = many_u16(338);
    level.tileWidth = one(322);
    level.tileHeight = one(323);
    level.tileOffsets = many(324);
    level.tileByteCounts = many(325);
    if (subIfds) {
        *subIfds = many(330);
    }
    return level;
}

void validate_tiff_level(const image_tiff::TiffLevel& level, const std::filesystem::path& path) {
    if (level.width == 0 || level.height == 0 || level.tileWidth == 0 || level.tileHeight == 0) {
        error("%s: TIFF must contain non-zero ImageWidth/ImageLength/TileWidth/TileLength: %s",
            __func__, path.string().c_str());
    }
    if (level.tileOffsets.empty() || level.tileByteCounts.empty()) {
        error("%s: only tiled TIFFs are supported; %s has no TileOffsets/TileByteCounts",
            __func__, path.string().c_str());
    }
    if (level.tileOffsets.size() != level.tileByteCounts.size()) {
        error("%s: TileOffsets and TileByteCounts length mismatch in %s",
            __func__, path.string().c_str());
    }
    if (!(level.compression == 1 || level.compression == 8 || level.compression == 32946)) {
        error("%s: unsupported TIFF compression %u in %s; convert to uncompressed or Deflate",
            __func__, static_cast<unsigned>(level.compression), path.string().c_str());
    }
    if (level.planarConfig != 1) {
        error("%s: only chunky TIFF planar configuration is supported: %s", __func__, path.string().c_str());
    }
    const bool gray = level.photometric == 0 || level.photometric == 1;
    const bool rgb = level.photometric == 2;
    if (!gray && !rgb) {
        error("%s: unsupported TIFF photometric interpretation %u in %s",
            __func__, static_cast<unsigned>(level.photometric), path.string().c_str());
    }
    if (rgb) {
        if (!(level.samplesPerPixel == 3 || level.samplesPerPixel == 4) ||
                !(level.bitsPerSample.size() == 3 || level.bitsPerSample.size() == 4)) {
            error("%s: RGB TIFF must have 3 or 4 samples per pixel: %s", __func__, path.string().c_str());
        }
        for (uint16_t b : level.bitsPerSample) {
            if (b != 8) {
                error("%s: RGB TIFF currently requires 8-bit samples: %s", __func__, path.string().c_str());
            }
        }
    } else {
        if (level.samplesPerPixel != 1 || level.bitsPerSample.size() != 1 ||
                !(level.bitsPerSample.front() == 8 || level.bitsPerSample.front() == 16)) {
            error("%s: grayscale TIFF must be 8-bit or 16-bit single-sample: %s",
                __func__, path.string().c_str());
        }
    }
    for (uint16_t sf : level.sampleFormat) {
        if (sf != 1) {
            error("%s: only unsigned integer TIFF SampleFormat is supported: %s",
                __func__, path.string().c_str());
        }
    }
    const uint64_t tilesX = (level.width + level.tileWidth - 1) / level.tileWidth;
    const uint64_t tilesY = (level.height + level.tileHeight - 1) / level.tileHeight;
    if (tilesX != 0 && tilesY > std::numeric_limits<uint64_t>::max() / tilesX) {
        error("%s: tile grid overflow in %s", __func__, path.string().c_str());
    }
    if (tilesX * tilesY != level.tileOffsets.size()) {
        error("%s: tile count mismatch in %s", __func__, path.string().c_str());
    }
}

bool inflate_exact_with_window_bits(const uint8_t* src, size_t srcSize,
    std::vector<uint8_t>& out, int windowBits) {
    if (srcSize > std::numeric_limits<uInt>::max() || out.size() > std::numeric_limits<uInt>::max()) {
        error("%s: TIFF Deflate tile exceeds zlib single-call limit", __func__);
    }
    std::fill(out.begin(), out.end(), uint8_t{0});
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src));
    zs.avail_in = static_cast<uInt>(srcSize);
    zs.next_out = reinterpret_cast<Bytef*>(out.data());
    zs.avail_out = static_cast<uInt>(out.size());
    if (inflateInit2(&zs, windowBits) != Z_OK) {
        return false;
    }
    const int ret = inflate(&zs, Z_FINISH);
    const bool ok = ret == Z_STREAM_END && zs.total_out == out.size();
    inflateEnd(&zs);
    return ok;
}

std::vector<uint8_t> zlib_decompress_exact(const uint8_t* src, size_t srcSize, size_t expectedSize) {
    std::vector<uint8_t> out(expectedSize);
    if (!inflate_exact_with_window_bits(src, srcSize, out, MAX_WBITS) &&
        !inflate_exact_with_window_bits(src, srcSize, out, -MAX_WBITS)) {
        error("%s: failed decompressing TIFF Deflate tile", __func__);
    }
    return out;
}

uint16_t quantile_u16(std::vector<uint16_t> values, double percentile) {
    if (values.empty()) {
        return 0;
    }
    percentile = std::clamp(percentile, 0.0, 100.0);
    const double pos = percentile / 100.0 * static_cast<double>(values.size() - 1);
    const size_t k = static_cast<size_t>(std::llround(pos));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
    return values[k];
}

} // namespace

image_tiff::TiffInfo image_tiff::parse_tiff_info(RandomAccessFile& file) {
    if (file.size() < 8) {
        error("%s: %s is too small to be a TIFF", __func__, file.path().string().c_str());
    }
    const std::vector<uint8_t> header = file.read(0, std::min<uint64_t>(16, file.size()));
    TiffInfo info;
    if (header[0] == 'I' && header[1] == 'I') {
        info.littleEndian = true;
    } else if (header[0] == 'M' && header[1] == 'M') {
        info.littleEndian = false;
    } else {
        error("%s: %s is not a TIFF file", __func__, file.path().string().c_str());
    }
    const uint16_t magic = read_u16(header, 2, info.littleEndian);
    uint64_t ifdOffset = 0;
    if (magic == 42) {
        info.bigTiff = false;
        ifdOffset = read_u32(header, 4, info.littleEndian);
    } else if (magic == 43) {
        info.bigTiff = true;
        const uint16_t offsetSize = read_u16(header, 4, info.littleEndian);
        if (offsetSize != 8) {
            error("%s: unsupported BigTIFF offset size %u", __func__, static_cast<unsigned>(offsetSize));
        }
        ifdOffset = read_u64(header, 8, info.littleEndian);
    } else {
        error("%s: unsupported TIFF magic %u in %s", __func__, static_cast<unsigned>(magic),
            file.path().string().c_str());
    }
    std::vector<uint64_t> subIfds;
    info.levels.push_back(parse_tiff_level(file, info, ifdOffset, &subIfds));
    validate_tiff_level(info.levels.back(), file.path());
    std::set<uint64_t> seen{ifdOffset};
    for (uint64_t subIfd : subIfds) {
        if (!seen.insert(subIfd).second) {
            continue;
        }
        info.levels.push_back(parse_tiff_level(file, info, subIfd, nullptr));
        validate_tiff_level(info.levels.back(), file.path());
    }
    return info;
}

std::vector<uint8_t> image_tiff::decode_tiff_tile(RandomAccessFile& file,
    const TiffLevel& level,
    size_t tileIdx,
    size_t tileBytes) {
    const uint64_t off = level.tileOffsets[tileIdx];
    const uint64_t n = level.tileByteCounts[tileIdx];
    if (off > file.size() || n > file.size() - off) {
        error("%s: TIFF tile points outside file: %s", __func__, file.path().string().c_str());
    }
    if (level.compression == 1) {
        if (n < tileBytes) {
            error("%s: uncompressed TIFF tile is shorter than expected in %s",
                __func__, file.path().string().c_str());
        }
        return file.read(off, tileBytes);
    }
    std::vector<uint8_t> compressed = file.read(off, static_cast<size_t>(n));
    return zlib_decompress_exact(compressed.data(), compressed.size(), tileBytes);
}

image_tiff::Gray16Bounds image_tiff::estimate_gray16_bounds(RandomAccessFile& file,
    const TiffLevel& level,
    size_t tileBytes,
    bool littleEndian,
    double lowPercentile,
    double highPercentile,
    double sampleFraction,
    int32_t sampleTiles,
    uint32_t seed) {
    const size_t totalTiles = level.tileOffsets.size();
    if (totalTiles == 0) {
        return Gray16Bounds{};
    }
    const bool scanAllTiles = sampleTiles == 0;
    const size_t fractionTiles = static_cast<size_t>(
        std::ceil(std::clamp(sampleFraction, 0.0, 1.0) * static_cast<double>(totalTiles)));
    const size_t minTiles = sampleTiles < 0 ? 0 : static_cast<size_t>(sampleTiles);
    const size_t targetTiles = scanAllTiles
        ? totalTiles
        : std::min(totalTiles, std::max<size_t>(1, std::max(fractionTiles, minTiles)));

    std::vector<size_t> indices(totalTiles);
    std::iota(indices.begin(), indices.end(), size_t{0});
    if (!scanAllTiles && targetTiles < totalTiles) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    const size_t candidateTiles = scanAllTiles
        ? totalTiles
        : std::min(totalTiles, std::max(targetTiles, targetTiles * size_t{4}));
    indices.resize(candidateTiles);

    if (!scanAllTiles && candidateTiles > targetTiles) {
        struct TileScore {
            size_t idx = 0;
            double density = 0.0;
            size_t order = 0;
        };
        std::vector<TileScore> scores;
        scores.reserve(indices.size());
        const size_t pixelsPerTile = tileBytes / 2u;
        for (size_t order = 0; order < indices.size(); ++order) {
            const std::vector<uint8_t> tile = decode_tiff_tile(file, level, indices[order], tileBytes);
            size_t nonZero = 0;
            for (size_t p = 0; p < pixelsPerTile; ++p) {
                const uint16_t v = read_u16(tile, p * 2u, littleEndian);
                if (v > 0) {
                    ++nonZero;
                }
            }
            scores.push_back(TileScore{
                indices[order],
                pixelsPerTile == 0 ? 0.0 : static_cast<double>(nonZero) / static_cast<double>(pixelsPerTile),
                order});
        }
        const bool hasDenseTile = std::any_of(scores.begin(), scores.end(), [](const TileScore& score) {
            return score.density > 0.0;
        });
        if (hasDenseTile) {
            std::stable_sort(scores.begin(), scores.end(), [](const TileScore& a, const TileScore& b) {
                return a.density > b.density;
            });
        } else {
            std::stable_sort(scores.begin(), scores.end(), [](const TileScore& a, const TileScore& b) {
                return a.order < b.order;
            });
        }
        indices.clear();
        for (size_t i = 0; i < targetTiles; ++i) {
            indices.push_back(scores[i].idx);
        }
    } else {
        indices.resize(targetTiles);
    }

    constexpr size_t kMaxSamplesPerTile = 4096;
    std::vector<uint16_t> values;
    values.reserve(indices.size() * std::min(kMaxSamplesPerTile, tileBytes / 2u));
    for (size_t idx : indices) {
        const std::vector<uint8_t> tile = decode_tiff_tile(file, level, idx, tileBytes);
        const size_t pixelsPerTile = tile.size() / 2u;
        const size_t step = std::max<size_t>(1, pixelsPerTile / kMaxSamplesPerTile);
        for (size_t p = 0; p < pixelsPerTile; p += step) {
            values.push_back(read_u16(tile, p * 2u, littleEndian));
        }
    }
    if (values.empty()) {
        return Gray16Bounds{};
    }
    Gray16Bounds bounds;
    bounds.low = quantile_u16(values, lowPercentile);
    bounds.high = quantile_u16(std::move(values), highPercentile);
    if (bounds.high <= bounds.low) {
        bounds.high = static_cast<uint16_t>(std::min<uint32_t>(
            static_cast<uint32_t>(bounds.low) + 1u,
            static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())));
        if (bounds.high <= bounds.low) {
            bounds.low = static_cast<uint16_t>(bounds.high - 1u);
        }
    }
    notice("%s: 16-bit grayscale TIFF display bounds from %zu tile sample(s): %u..%u",
        __func__, indices.size(), static_cast<unsigned>(bounds.low), static_cast<unsigned>(bounds.high));
    return bounds;
}
