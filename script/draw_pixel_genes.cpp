#include "punkst.h"
#include "tilereader.hpp"
#include "utils.h"
#include "threads.hpp"
#include "image_utils.hpp"
#include <unordered_map>
#include <thread>
#include <mutex>
#include <vector>

struct TileResult {
    int top;
    int left;
    Image2D<Color3f> color_accumulator;
    Image2D<float> weight_accumulator;
    float median_weight;
};

void draw_worker(
    ThreadSafeQueue<TileInfo>& tileQueue,
    std::vector<TileResult>& results,
    std::mutex& results_mutex,
    const TileReader& tileReader,
    const lineParserUnival& parser,
    const std::unordered_map<uint32_t, std::vector<int32_t>>& feature_color_map,
    double xmin, double ymin, double scale,
    int image_width, int image_height
) {
    TileInfo block;
    while (tileQueue.pop(block)) {
        const double block_xmin = block.idx.xmin;
        const double block_ymin = block.idx.ymin;
        const double block_xmax = block.idx.xmax;
        const double block_ymax = block.idx.ymax;

        int left = static_cast<int>(std::floor((block_xmin - xmin) / scale));
        int top = static_cast<int>(std::floor((block_ymin - ymin) / scale));
        int right = static_cast<int>(std::ceil((block_xmax - xmin) / scale));
        int bottom = static_cast<int>(std::ceil((block_ymax - ymin) / scale));

        int tile_w = right - left;
        int tile_h = bottom - top;

        if (tile_w <= 0 || tile_h <= 0) continue;

        TileResult result;
        result.top = top;
        result.left = left;
        result.color_accumulator = Image2D<Color3f>(tile_h, tile_w, Color3f{0.f, 0.f, 0.f});
        result.weight_accumulator = Image2D<float>(tile_h, tile_w, 0.f);

        auto iter = tileReader.get_block_iterator(block);
        if (!iter) continue;

        std::string line;
        while (iter->next(line)) {
            RecordT<float> rec;
            if (parser.parse(rec, line, true) < 0 || rec.ct <= 0) continue;

            int xpix = static_cast<int>((rec.x - xmin) / scale) - left;
            int ypix = static_cast<int>((rec.y - ymin) / scale) - top;

            if (xpix < 0 || xpix >= tile_w || ypix < 0 || ypix >= tile_h) continue;

            auto it = feature_color_map.find(rec.idx);
            if (it != feature_color_map.end()) {
                const auto& color = it->second;
                float weight = static_cast<float>(rec.ct);
                result.color_accumulator(ypix, xpix) += Color3f(color[0] * weight, color[1] * weight, color[2] * weight);
                result.weight_accumulator(ypix, xpix) += weight;
            }
        }

        std::vector<float> non_zero_weights;
        for (int r = 0; r < result.weight_accumulator.height(); ++r) {
            for (int c = 0; c < result.weight_accumulator.width(); ++c) {
                float w = result.weight_accumulator(r, c);
                if (w > 0) {
                    non_zero_weights.push_back(w);
                }
            }
        }

        if (non_zero_weights.empty()) {
            result.median_weight = 0;
        } else {
            std::sort(non_zero_weights.begin(), non_zero_weights.end());
            if (non_zero_weights.size() % 2 == 0) {
                result.median_weight = (non_zero_weights[non_zero_weights.size() / 2 - 1] + non_zero_weights[non_zero_weights.size() / 2]) / 2.0f;
            } else {
                result.median_weight = non_zero_weights[non_zero_weights.size() / 2];
            }
        }

        {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(std::move(result));
        }
    }
}

int32_t cmdDrawPixelFeatures(int32_t argc, char** argv) {
    std::string inPrefix, dataFile, indexFile, featureColorFile, dictFile, rangeFile, outFile;
    std::vector<std::string> featureListStr, colorListStr;
    double scale = 1.0;
    double xmin = 0, xmax = -1, ymin = 0, ymax = -1;
    int32_t verbose = 1000000;
    int icol_x, icol_y, icol_feature, icol_val;
    int n_threads = 1;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV file from pts2tiles.", dataFile)
      .add_option("in-data", "Input TSV file from pts2tiles.", dataFile)
      .add_option("in-index", "Index file for the TSV.", indexFile)
      .add_option("in", "Input prefix (equal to --in-data <in>.tsv --in-index <in>.index)", inPrefix)
      .add_option("icol-x", "0-based column for x-coordinate.", icol_x, true)
      .add_option("icol-y", "0-based column for y-coordinate.", icol_y, true)
      .add_option("icol-feature", "0-based column for feature ID/name.", icol_feature, true)
      .add_option("icol-val", "0-based column for feature value.", icol_val, true)
      .add_option("feature-dict", "Dictionary file to map feature names to integer IDs.", dictFile)
      .add_option("feature-color-map", "TSV file with feature names and hex colors.", featureColorFile)
      .add_option("feature-list", "A list of feature names to draw.", featureListStr)
      .add_option("color-list", "A list of hex colors (#RRGGBB) for features.", colorListStr)
      .add_option("scale", "Scale factor for coordinates.", scale)
      .add_option("range", "File with coordinate range (xmin ymin xmax ymax).", rangeFile)
      .add_option("xmin", "Minimum x coordinate.", xmin)
      .add_option("xmax", "Maximum x coordinate.", xmax)
      .add_option("ymin", "Minimum y coordinate.", ymin)
      .add_option("ymax", "Maximum y coordinate.", ymax)
      .add_option("threads", "Number of threads to use.", n_threads);
    pl.add_option("out", "Output image file.", outFile, true)
      .add_option("verbose", "Frequency of progress messages.", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (!checkOutputWritable(outFile))
        error("Output file is not writable: %s", outFile.c_str());
    if (!inPrefix.empty()) {
        dataFile = inPrefix + ".tsv";
        indexFile = inPrefix + ".index";
    } else if (dataFile.empty()) {
        error("One of --in or --in-tsv & --in-data pair must be specified");
    }

    if (!rangeFile.empty()) {
        readCoordRange(rangeFile, xmin, xmax, ymin, ymax);
    }
    const bool manualRangeSpecified = xmin < xmax && ymin < ymax;

    TileReader metaReader(dataFile, indexFile);
    if (!metaReader.isValid()) {
        error("Failed to initialize TileReader. Check input TSV and index files.");
    }
    if (!manualRangeSpecified) {
        if (metaReader.hasGlobalBox()) {
            const auto& globalBox = metaReader.getGlobalBox();
            xmin = globalBox.xmin;
            xmax = globalBox.xmax;
            ymin = globalBox.ymin;
            ymax = globalBox.ymax;
            notice("Using coordinate range from index: xmin=%.2f xmax=%.2f ymin=%.2f ymax=%.2f", xmin, xmax, ymin, ymax);
        } else {
            error("Please set valid --range or x/y min/max options if the index does not contain global box information.");
        }
    }

    std::vector<Rectangle<double>> query_rects;
    query_rects.emplace_back(xmin, ymin, xmax, ymax);

    TileReader tileReader(dataFile, indexFile, &query_rects);
    if (!tileReader.isValid()) {
        error("Failed to initialize the data loader, check input TSV and index files.");
    }
    if (tileReader.getNumBlocks() == 0) {
        notice("No data overlap the requested rectangle: xmin=%.2f xmax=%.2f ymin=%.2f ymax=%.2f",
              xmin, xmax, ymin, ymax);
    }

    std::unordered_map<uint32_t, std::vector<int32_t>> feature_color_map;
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile, &query_rects);

    if (dictFile.empty()) {
        if (!featureColorFile.empty()) {
            std::vector<std::string> requested_features;
            std::ifstream fc_file(featureColorFile);
            if (!fc_file) error("Cannot open feature-color-map file: %s", featureColorFile.c_str());
            std::string line;
            while (std::getline(fc_file, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::vector<std::string> tokens;
                split(tokens, "\t", line);
                if (!tokens.empty()) {
                    requested_features.push_back(tokens[0]);
                }
            }
            parser.setFeatureDict(requested_features);
        } else {
            parser.setFeatureDict(featureListStr);
        }
    }

    if (!featureColorFile.empty()) {
        std::ifstream fc_file(featureColorFile);
        if (!fc_file) error("Cannot open feature-color-map file: %s", featureColorFile.c_str());
        std::string line;
        while (std::getline(fc_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < 2) continue;
            auto it = parser.featureDict.find(tokens[0]);
            if (it != parser.featureDict.end()) {
                std::vector<int32_t> rgb;
                if (set_rgb(tokens[1].c_str(), rgb)) {
                    feature_color_map[it->second] = rgb;
                } else {
                    warning("Invalid color format for feature %s: %s", tokens[0].c_str(), tokens[1].c_str());
                }
            }
        }
        notice("Loaded %zu colors from %s", feature_color_map.size(), featureColorFile.c_str());
    } else if (!featureListStr.empty()) {
        if (colorListStr.empty())
            colorListStr = std::vector<std::string>{"FFEE00", "DD65E6", "00FFFF", "FF7000"};
        if (featureListStr.size()>colorListStr.size())
            error("--color-list must have at least the same number of elements as --feature-list does");
        for (size_t i = 0; i < featureListStr.size(); ++i) {
            auto it = parser.featureDict.find(featureListStr[i]);
            if (it != parser.featureDict.end()) {
                std::vector<int32_t> rgb;
                if (set_rgb(colorListStr[i].c_str(), rgb)) {
                    feature_color_map[it->second] = rgb;
                } else {
                    warning("Invalid color format for feature %s: %s", featureListStr[i].c_str(), colorListStr[i].c_str());
                }
            }
        }
        notice("Loaded %zu colors from command line options.", feature_color_map.size());
    } else {
        error("No color information provided. Use --feature-color-map or --feature-list/--color-list.");
    }

    int width = static_cast<int>(std::ceil((xmax - xmin) / scale));
    int height = static_cast<int>(std::ceil((ymax - ymin) / scale));
    if (width <= 0 || height <= 0)
        error("Image dimensions are non-positive. Check range and scale. W=%d, H=%d", width, height);
    notice("Output image size: %d x %d", width, height);

    std::vector<TileInfo> blocks;
    tileReader.getBlockList(blocks);
    notice("Processing %zu indexed blocks overlapping the query rectangle with %d threads...", blocks.size(), n_threads);

    ThreadSafeQueue<TileInfo> tileQueue;
    for (const auto& block : blocks) {
        tileQueue.push(block);
    }
    tileQueue.set_done();

    std::vector<std::thread> threads;
    std::vector<TileResult> results;
    std::mutex results_mutex;

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(draw_worker, std::ref(tileQueue),
            std::ref(results), std::ref(results_mutex),
            std::cref(tileReader), std::cref(parser),
            std::cref(feature_color_map),
            xmin, ymin, scale, width, height);
    }

    for (auto& th : threads) {
        th.join();
    }

    notice("Finished processing tiles. Assembling final image from %zu tile results...", results.size());

    Image2D<Color3f> global_color_accumulator(height, width, Color3f{0.f, 0.f, 0.f});
    Image2D<float> global_weight_accumulator(height, width, 0.f);
    IntRect global_rect{0, 0, width, height};
    std::vector<float> per_tile_medians;

    for (const auto& res : results) {
        IntRect tile_rect{res.left, res.top, res.color_accumulator.width(), res.color_accumulator.height()};
        IntRect roi = intersect_rect(global_rect, tile_rect);
        if (roi.width > 0 && roi.height > 0) {
            const int src_x0 = roi.x - res.left;
            const int src_y0 = roi.y - res.top;
            for (int yy = 0; yy < roi.height; ++yy) {
                for (int xx = 0; xx < roi.width; ++xx) {
                    global_color_accumulator(roi.y + yy, roi.x + xx) +=
                        res.color_accumulator(src_y0 + yy, src_x0 + xx);
                    global_weight_accumulator(roi.y + yy, roi.x + xx) +=
                        res.weight_accumulator(src_y0 + yy, src_x0 + xx);
                }
            }
        }
        if (res.median_weight > 0) {
            per_tile_medians.push_back(res.median_weight);
        }
    }

    float global_median_weight = 0;
    if (!per_tile_medians.empty()) {
        std::sort(per_tile_medians.begin(), per_tile_medians.end());
        if (per_tile_medians.size() % 2 == 0) {
            global_median_weight = (per_tile_medians[per_tile_medians.size() / 2 - 1] + per_tile_medians[per_tile_medians.size() / 2]) / 2.0f;
        } else {
            global_median_weight = per_tile_medians[per_tile_medians.size() / 2];
        }
    }
    notice("Global median weight calculated: %.2f", global_median_weight);

    Image2D<Rgb8> out_image(height, width, Rgb8{0, 0, 0});
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float total_weight = global_weight_accumulator(y, x);
            if (total_weight > 0) {
                Color3f avg_color = global_color_accumulator(y, x) / total_weight;
                float intensity = (global_median_weight > 0) ? std::min(1.0f, total_weight / global_median_weight) : 0.0f;
                Color3f final_color = avg_color * intensity;
                out_image(y, x) = Rgb8{clamp_u8(final_color.r),
                                        clamp_u8(final_color.g),
                                        clamp_u8(final_color.b)};
            }
        }
    }

    notice("Writing image to %s", outFile.c_str());
    save_png_rgb8(outFile, out_image);

    return 0;
}
