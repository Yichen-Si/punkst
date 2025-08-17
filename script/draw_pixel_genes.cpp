#include "tilereader.hpp"
#include "utils.h"
#include "threads.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <vector>

struct TileResult {
    int top;
    int left;
    cv::Mat3f color_accumulator;
    cv::Mat1f weight_accumulator;
    float median_weight;
};

void draw_worker(
    ThreadSafeQueue<TileKey>& tileQueue,
    std::vector<TileResult>& results,
    std::mutex& results_mutex,
    const TileReader& tileReader,
    const lineParserUnival& parser,
    const std::unordered_map<uint32_t, std::vector<int32_t>>& feature_color_map,
    double xmin, double ymin, double scale,
    int image_width, int image_height
) {
    TileKey tile_key;
    while (tileQueue.pop(tile_key)) {
        int tile_size = tileReader.getTileSize();
        double tile_xmin = tile_key.col * tile_size;
        double tile_ymin = tile_key.row * tile_size;

        int left = static_cast<int>(std::floor((tile_xmin - xmin) / scale));
        int top = static_cast<int>(std::floor((tile_ymin - ymin) / scale));
        int right = static_cast<int>(std::ceil(((tile_xmin + tile_size) - xmin) / scale));
        int bottom = static_cast<int>(std::ceil(((tile_ymin + tile_size) - ymin) / scale));

        int tile_w = right - left;
        int tile_h = bottom - top;

        if (tile_w <= 0 || tile_h <= 0) continue;

        TileResult result;
        result.top = top;
        result.left = left;
        result.color_accumulator = cv::Mat3f::zeros(tile_h, tile_w);
        result.weight_accumulator = cv::Mat1f::zeros(tile_h, tile_w);

        auto iter = tileReader.get_tile_iterator(tile_key.row, tile_key.col);
        if (!iter) continue;

        std::string line;
        while (iter->next(line)) {
            RecordT<double> rec;
            if (parser.parse(rec, line) < 0 || rec.ct <= 0) continue;

            int xpix = static_cast<int>((rec.x - xmin) / scale) - left;
            int ypix = static_cast<int>((rec.y - ymin) / scale) - top;

            if (xpix < 0 || xpix >= tile_w || ypix < 0 || ypix >= tile_h) continue;

            auto it = feature_color_map.find(rec.idx);
            if (it != feature_color_map.end()) {
                const auto& color = it->second; // BGR
                float weight = static_cast<float>(rec.ct);
                result.color_accumulator(ypix, xpix) += cv::Vec3f(color[0] * weight, color[1] * weight, color[2] * weight);
                result.weight_accumulator(ypix, xpix) += weight;
            }
        }

        std::vector<float> non_zero_weights;
        for (int r = 0; r < result.weight_accumulator.rows; ++r) {
            for (int c = 0; c < result.weight_accumulator.cols; ++c) {
                float w = result.weight_accumulator.at<float>(r, c);
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
    std::string dataFile, indexFile, featureColorFile, dictFile, rangeFile, outFile;
    std::vector<std::string> featureListStr, colorListStr;
    double scale = 1.0;
    double xmin = 0, xmax = -1, ymin = 0, ymax = -1;
    int32_t verbose = 1000000;
    int icol_x, icol_y, icol_feature, icol_val;
    int n_threads = 1;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV file from pts2tiles.", dataFile, true)
      .add_option("in-index", "Index file for the TSV.", indexFile, true)
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
        pl.print_help();
        return 1;
    }

    if (!checkOutputWritable(outFile))
        error("Output file is not writable: %s", outFile.c_str());

    if (!rangeFile.empty()) {
        readCoordRange(rangeFile, xmin, xmax, ymin, ymax);
    }
    if (xmin >= xmax || ymin >= ymax)
        error("Invalid range: xmin >= xmax or ymin >= ymax. Please set --range or x/y min/max options.");

    std::unordered_map<uint32_t, std::vector<int32_t>> feature_color_map;
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile);

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
                std::vector<int32_t> bgr;
                if (set_rgb(tokens[1].c_str(), bgr)) {
                    feature_color_map[it->second] = bgr;
                } else {
                    warning("Invalid color format for feature %s: %s", tokens[0].c_str(), tokens[1].c_str());
                }
            }
        }
        notice("Loaded %zu colors from %s", feature_color_map.size(), featureColorFile.c_str());
    } else if (!featureListStr.empty()) {
        if (featureListStr.size() != colorListStr.size())
            error("--feature-list and --color-list must have the same number of elements.");
        for (size_t i = 0; i < featureListStr.size(); ++i) {
            auto it = parser.featureDict.find(featureListStr[i]);
            if (it != parser.featureDict.end()) {
                std::vector<int32_t> bgr;
                if (set_rgb(colorListStr[i].c_str(), bgr)) {
                    feature_color_map[it->second] = bgr;
                } else {
                    warning("Invalid color format for feature %s: %s", featureListStr[i].c_str(), colorListStr[i].c_str());
                }
            }
        }
        notice("Loaded %zu colors from command line options.", feature_color_map.size());
    } else {
        error("No color information provided. Use --feature-color-map or --feature-list/--color-list.");
    }

    TileReader tileReader(dataFile, indexFile);
    if (!tileReader.isValid()) {
        error("Failed to initialize TileReader. Check input TSV and index files.");
    }

    int width = static_cast<int>(std::ceil((xmax - xmin) / scale));
    int height = static_cast<int>(std::ceil((ymax - ymin) / scale));
    if (width <= 0 || height <= 0)
        error("Image dimensions are non-positive. Check range and scale. W=%d, H=%d", width, height);
    notice("Output image size: %d x %d", width, height);

    std::vector<TileKey> tiles;
    tileReader.getTileList(tiles);
    notice("Processing %zu tiles with %d threads...", tiles.size(), n_threads);

    ThreadSafeQueue<TileKey> tileQueue;
    for (const auto& tile_key : tiles) {
        tileQueue.push(tile_key);
    }
    tileQueue.set_done();

    std::vector<std::thread> threads;
    std::vector<TileResult> results;
    std::mutex results_mutex;

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(draw_worker, std::ref(tileQueue), std::ref(results), std::ref(results_mutex),
                               std::cref(tileReader), std::cref(parser), std::cref(feature_color_map),
                               xmin, ymin, scale, width, height);
    }

    for (auto& th : threads) {
        th.join();
    }

    notice("Finished processing tiles. Assembling final image from %zu tile results...", results.size());

    cv::Mat3f global_color_accumulator = cv::Mat3f::zeros(height, width);
    cv::Mat1f global_weight_accumulator = cv::Mat1f::zeros(height, width);
    cv::Rect global_rect(0, 0, width, height);
    std::vector<float> per_tile_medians;

    for (const auto& res : results) {
        cv::Rect tile_rect(res.left, res.top, res.color_accumulator.cols, res.color_accumulator.rows);
        cv::Rect roi = global_rect & tile_rect;
        if (roi.width > 0 && roi.height > 0) {
            cv::Rect src_rect(roi.x - res.left, roi.y - res.top, roi.width, roi.height);
            global_color_accumulator(roi) += res.color_accumulator(src_rect);
            global_weight_accumulator(roi) += res.weight_accumulator(src_rect);
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

    cv::Mat out_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float total_weight = global_weight_accumulator(y, x);
            if (total_weight > 0) {
                cv::Vec3f avg_color = global_color_accumulator(y, x) / total_weight;
                float intensity = (global_median_weight > 0) ? std::min(1.0f, total_weight / global_median_weight) : 0.0f;
                cv::Vec3f final_color = avg_color * intensity;
                out_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    cv::saturate_cast<uchar>(final_color[2]), // B
                    cv::saturate_cast<uchar>(final_color[1]), // G
                    cv::saturate_cast<uchar>(final_color[0])  // R
                );
            }
        }
    }

    notice("Writing image to %s", outFile.c_str());
    if (!cv::imwrite(outFile, out_image)) {
        error("Failed to write output image to %s", outFile.c_str());
    }

    return 0;
}
