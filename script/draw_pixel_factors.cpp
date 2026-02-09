#include "punkst.h"
#include "utils_sys.hpp"
#include "tileoperator.hpp"
#include "dataunits.hpp"
#include "img_utils.hpp"
#include <opencv2/opencv.hpp>

int32_t cmdDrawPixelFactors(int32_t argc, char** argv) {
    std::string dataFile, inPrefix, indexFile, headerFile, rangeFile, colorFile, outFile;
    std::vector<std::string> channelListStr, colorListStr;
    double scale = 1;
    float xmin = 0, xmax = -1, ymin = 0, ymax = -1;
    int32_t verbose = 1000000;
    bool filter = false;
    bool topOnly = false;
    bool isBinary = false;
    float minProb = 1e-3;
    int32_t debug_ = 0;
    int32_t islandSmooth = 0;
    bool fillEmptyIslands = false;

    ParamList pl;
    // Input options
    pl.add_option("in-data", "Input data file. Lines begin with # will be ignored", dataFile)
      .add_option("index", "Index file", indexFile)
      .add_option("in", "Input prefix (equal to --in-tsv <in>.tsv/.bin --index <in>.index)", inPrefix)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("in-tsv", "Input TSV file. Lines begin with # will be ignored", dataFile) // backward compatible
      .add_option("header-json", "Header JSON file", headerFile) // to deprecate
      .add_option("in-color", "Input color file (RGB triples)", colorFile)
      .add_option("scale", "Scale factor: (x-xmin)/scale → pixel_x", scale)
      .add_option("range", "A file containing coordinate range (xmin ymin xmax ymax)", rangeFile)
      .add_option("xmin", "Minimum x coordinate", xmin)
      .add_option("xmax", "Maximum x coordinate", xmax)
      .add_option("ymin", "Minimum y coordinate", ymin)
      .add_option("ymax", "Maximum y coordinate", ymax)
      .add_option("filter", "Access only the queried region using the index", filter)
      .add_option("channel-list", "A list of channel IDs to draw", channelListStr)
      .add_option("color-list", "A list of colors for channels in hex code (#RRGGBB)", colorListStr)
      .add_option("min-prob", "Minimum probability to consider a pixel", minProb);
    // Output
    pl.add_option("out", "Output image file", outFile, true)
      .add_option("top-only", "Use only the top channel per pixel", topOnly)
      .add_option("island-smooth", "Remove isolated noisy pixels (only for --top-only)", islandSmooth)
      .add_option("fill-empty-islands", "Fill empty pixels surrounded by consistent neighbors (only for --island-smooth)", fillEmptyIslands)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (debug_ > 0) {
        logger::Logger::getInstance().setLevel(logger::LogLevel::DEBUG);
    }

    if (!checkOutputWritable(outFile))
        error("Output file is not writable: %s", outFile.c_str());

    if (!inPrefix.empty()) {
        dataFile = inPrefix + (isBinary ? ".bin" : ".tsv");
        indexFile = inPrefix + ".index";
    } else if (dataFile.empty()) {
        error("One of --in --in-tsv or --in-data must be specified");
    }
    if (filter && indexFile.empty())
        error("Index file is required when --filter is set");

    // set up reader
    TileOperator reader(dataFile, indexFile, headerFile);
    int32_t k = reader.getK();
    if (k<=0) error("No factor columns found in header");
    if (!rangeFile.empty()) {
       readCoordRange(rangeFile, xmin, xmax, ymin, ymax);
    }
    if (xmin >= xmax || ymin >= ymax) {
        if (!indexFile.empty() && reader.getBoundingBox(xmin, xmax, ymin, ymax)) {
            notice("Using full data range from index: xmin=%.1f, xmax=%.1f, ymin=%.1f, ymax=%.1f", xmin, xmax, ymin, ymax);
        } else {
            error("Invalid range: xmin >= xmax or ymin >= ymax");
        }
    }
    if (filter) {
        int32_t ntiles = reader.query(xmin, xmax, ymin, ymax);
        if (ntiles <= 0)
            error("No data in the queried region");
        notice("Found %d tiles intersecting the queried region", ntiles);
    } else {
        reader.openDataStream();
    }

    bool selected = !channelListStr.empty();
    if (!selected && colorFile.empty())
        error("Either --in-color or both --channel-list and --color-list must be provided");

    std::vector<std::vector<int>> cmtx;
    std::unordered_map<int,std::vector<int32_t>> selectedMap;
    if (selected) { // parse selected channels
        if (colorListStr.empty()) {
            colorListStr = std::vector<std::string>{"FFEE00", "DD65E6", "00FFFF", "FF7000"};
        }
        if (channelListStr.size()>colorListStr.size()) {
            error("--channel-list and --color-list must have same length");
        }
        for (size_t i=0;i<channelListStr.size();++i) {
            std::vector<int32_t> c;
            if (!set_rgb(colorListStr[i].c_str(), c))
                error("Invalid --color-list");
            selectedMap[ std::stoi(channelListStr[i]) ] = c;
        }
    } else {
        std::ifstream cs(colorFile);
        if (!cs.is_open())
            error("Error opening color file: %s", colorFile.c_str());
        std::string line;
        while (std::getline(cs,line)) {
            if (line.empty() || line[0] == '#' || line[0] == 'R') continue;
            std::istringstream iss(line);
            int r,g,b;
            if (iss>>r>>g>>b) cmtx.push_back({r,g,b});
        }
        notice("Loaded %zu colors from %s", cmtx.size(), colorFile.c_str());
    }
    int ncolor = (int) cmtx.size();

    if (scale<=0) error("--scale must be >0");

    // image dims
    int width  = int(std::floor((xmax-xmin)/scale))+1;
    int height = int(std::floor((ymax-ymin)/scale))+1;
    notice("Image size: %d x %d", width, height);
    if (width<=1||height<=1)
        error("Image dimensions are zero; check your bounds/scale");

    if (topOnly && islandSmooth > 0) {
        cv::Mat1i topCh(height, width, int(-1)); // store the top assignment
        PixTopProbs<float> rec;
        int32_t ret, nline=0, nskip=0, nkept=0;
        // 1) Read and record top label per pixel
        struct SmallCounts {
            uint8_t m = 0;
            int32_t lab[8];
            uint8_t cnt[8];
            inline void add(int32_t v) {
                for (uint8_t i=0; i<m; ++i) {
                    if (lab[i] == v) { ++cnt[i]; return; }
                }
                if (m < 8) { lab[m] = v; cnt[m] = 1; ++m; }
                else {
                    // unlikely; ignore further distinct labels
                }
            }
        };
        cv::Mat1b cnt0(height, width, uchar(0));
        cv::Mat1b tot(height, width, uchar(0));
        std::unordered_map<int, SmallCounts> extra; // only pixels with collisions end up here
        std::vector<int> coords;
        while ((ret = reader.next(rec)) >= 0) {
            if (ret==0) {
                if (nkept>10000) {
                    warning("Stopped at invalid line %d", nline);
                    break;
                }
                error("%s: Invalid or corrupted input", __FUNCTION__);
            }
            if (++nline % verbose == 0)
                notice("Processed %d lines, skipped %d, kept %d", nline, nskip, nkept);
            int xpix = int((rec.x - xmin)/scale);
            int ypix = int((rec.y - ymin)/scale);
            if ((unsigned)xpix >= (unsigned)width || (unsigned)ypix >= (unsigned)height) continue;
            if (tot(ypix, xpix) >= 255) { ++nskip; continue; }
            // map record -> integer label
            int ch = rec.ks[0];
            int lbl = -1;
            if (selected) {
                auto it = selectedMap.find(ch);
                if (it == selectedMap.end()) continue;
                lbl = ch; // store channel id
            } else {
                if (ch < 0 || ch >= (int)cmtx.size()) ch = ((ch % ncolor) + ncolor) % ncolor;
                lbl = ch; // store color index
            }

            int idx = ypix*width + xpix;
            // first time this pixel appears
            if (topCh(ypix, xpix) == -1) {
                topCh(ypix, xpix) = lbl;
                cnt0(ypix, xpix) = 1;
                tot(ypix, xpix)  = 1;
                coords.push_back(idx);
                ++nkept;
                continue;
            }
            // subsequent hits
            ++tot(ypix, xpix);
            if (lbl == topCh(ypix, xpix)) {
                ++cnt0(ypix, xpix);
            } else {
                extra[idx].add(lbl);
            }
            ++nkept;
        }

        // --- finalize per-pixel mode (exact) ---
        for (int idx : coords) {
            int y = idx / width;
            int x = idx - y*width;
            int bestLbl = topCh(y,x);
            int bestCnt = cnt0(y,x);
            auto it = extra.find(idx);
            if (it != extra.end()) {
                const SmallCounts &sc = it->second;
                for (uint8_t i=0; i<sc.m; ++i) {
                    if (sc.cnt[i] > bestCnt) {
                        bestCnt = sc.cnt[i];
                        bestLbl = sc.lab[i];
                    }
                    // tie policy: keep existing bestLbl if equal (stable)
                }
            }
            topCh(y,x) = bestLbl; // overwrite with final mode label
        }
        notice("Finished reading input, start smoothing");

        // 2) Smooth out isolated noise
        std::vector<int32_t> smLabels(static_cast<size_t>(width) * static_cast<size_t>(height), -1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                smLabels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = topCh(y, x);
            }
        }
        island_smoothing::Options smoothOpts;
        smoothOpts.fillEmpty = fillEmptyIslands;
        smoothOpts.updateProbFromWinnerMin = false;
        island_smoothing::Result smoothRes = island_smoothing::smoothLabels8Neighborhood(
            smLabels, nullptr, static_cast<size_t>(width), static_cast<size_t>(height), islandSmooth, smoothOpts);
        if (smoothRes.converged) {
            notice("Island smoothing converged after %d rounds", smoothRes.roundsRun);
        } else {
            notice("Finished round %d of island smoothing", islandSmooth);
        }

        // 3) Render RGB from smoothed labels
        cv::Mat out(height, width, CV_8UC3, cv::Scalar(0,0,0));

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int lbl = smLabels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
                if (lbl == -1) continue;

                int R=0, G=0, B=0;
                if (selected) {
                    auto it = selectedMap.find(lbl);
                    if (it == selectedMap.end()) continue; // should not happen
                    const auto &c = it->second;
                    R = c[0]; G = c[1]; B = c[2];
                } else {
                    // lbl already in [0, ncolor)
                    R = cmtx[lbl][0];
                    G = cmtx[lbl][1];
                    B = cmtx[lbl][2];
                }

                out.at<cv::Vec3b>(y,x) = cv::Vec3b(
                    cv::saturate_cast<uchar>(B),
                    cv::saturate_cast<uchar>(G),
                    cv::saturate_cast<uchar>(R)
                );
            }
        }

        notice("Writing image to %s ...", outFile.c_str());
        if (!cv::imwrite(outFile, out))
            error("Error writing output image: %s", outFile.c_str());

        return 0;
    }

    // accumulators
    cv::Mat3f sumImg(height, width, cv::Vec3f(0,0,0));
    cv::Mat1b countImg(height, width, uchar(0));
    // read & accumulate
    PixTopProbs<float> rec;
    int32_t ret, nline=0, nskip=0, nkept=0;
    while ((ret = reader.next(rec)) >= 0) {
        if (ret==0) {
            if (nkept>10000) {
                warning("Stopped at invalid line %d", nline);
                break;
            }
            error("%s: Invalid or corrupted input", __FUNCTION__);
        }
        if (++nline % verbose == 0)
            notice("Processed %d lines, skipped %d, kept %d", nline, nskip, nkept);

        int xpix = int((rec.x - xmin)/scale);
        int ypix = int((rec.y - ymin)/scale);
        if (xpix<0||xpix>=width||ypix<0||ypix>=height) {
            debug("Skipping out-of-bounds pixel (%.1f, %.1f) → (%d, %d)", rec.x, rec.y, xpix, ypix);
            if (debug_ && nline > debug_) {
                return 0;
            }
            continue;
        }
        if (countImg(ypix, xpix)>=255) { nskip++; continue; }

        float R=0,G=0,B=0;
        if (topOnly || k==1) {
            int ch = rec.ks[0];
            if (selected) {
                auto it = selectedMap.find(ch);
                if (it == selectedMap.end()) { continue; }
                auto& c = it->second;
                R = c[0];
                G = c[1];
                B = c[2];
            } else {
                if (ch<0 && ch>=(int)cmtx.size()) {
                    // debug("Channel index out of range: %d", ch);
                    // continue;
                    ch = ch % ncolor;
                }
                R = cmtx[ch][0];
                G = cmtx[ch][1];
                B = cmtx[ch][2];
            }
            sumImg(ypix, xpix) += cv::Vec3f(R,G,B);
            countImg(ypix, xpix) += 1;
            ++nkept;
            continue;
        }

        bool valid=false;
        double psum=0;
        for (int i=0;i<k;++i) {
            int ch = rec.ks[i];
            double p = rec.ps[i];
            if (p < minProb) continue;
            if (selected) {
                auto it = selectedMap.find(ch);
                if (it == selectedMap.end()) continue;
                auto& c = it->second;
                R += c[0]*p;
                G += c[1]*p;
                B += c[2]*p;
                valid = true;
            } else {
                if (ch<0 || ch>= (int)cmtx.size()) {
                    // warning("Channel index out of range: %d", ch);
                    // continue;
                    ch = ch % ncolor;
                }
                R += cmtx[ch][0]*p;
                G += cmtx[ch][1]*p;
                B += cmtx[ch][2]*p;
                valid = true;
            }
            psum += p;
        }
        if (!valid || psum < 1e-3) {
            debug("Skipping pixel with no valid channels at (%.1f, %.1f) (psum=%.1e)", rec.x, rec.y, psum);
            continue;
        }
        R /= psum; G /= psum; B /= psum;
        sumImg(ypix, xpix) += cv::Vec3f(R,G,B);
        countImg(ypix, xpix) += 1;
        ++nkept;
    }
    notice("Finished reading input; building image");

    // finalize image
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(0,0,0));
    for (int y=0;y<height;++y) {
        for (int x=0;x<width;++x) {
            if (countImg(y,x)) {
                cv::Vec3f avg = sumImg(y,x) / countImg(y,x);
                out.at<cv::Vec3b>(y,x) = cv::Vec3b(
                    cv::saturate_cast<uchar>(avg[2]),  // B
                    cv::saturate_cast<uchar>(avg[1]),  // G
                    cv::saturate_cast<uchar>(avg[0])   // R
                );
            }
        }
    }

    notice("Writing image to %s ...", outFile.c_str());
    if (!cv::imwrite(outFile, out))
        error("Error writing output image: %s", outFile.c_str());

    return 0;
}
