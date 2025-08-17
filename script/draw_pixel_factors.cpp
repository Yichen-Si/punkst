#include "pixresultreader.hpp"
#include "dataunits.hpp"
#include <opencv2/opencv.hpp>

int32_t cmdDrawPixelFactors(int32_t argc, char** argv) {
    std::string dataFile, indexFile, headerFile, rangeFile, colorFile, outFile;
    std::vector<std::string> channelListStr, colorListStr;
    double scale = 1;
    double xmin = 0, xmax = -1, ymin = 0, ymax = -1;
    int32_t verbose = 1000000;
    bool filter = false;

    ParamList pl;
    // Input options
    pl.add_option("in-tsv", "Input TSV file. Lines begin with # will be ignored", dataFile, true)
      .add_option("header-json", "Header JSON file", headerFile)
      .add_option("index", "Index file", indexFile)
      .add_option("in-color", "Input color file (RGB triples)", colorFile)
      .add_option("scale", "Scale factor: (x-xmin)/scale → pixel_x", scale)
      .add_option("range", "A file containing coordinate range (xmin ymin xmax ymax)", rangeFile)
      .add_option("xmin", "Minimum x coordinate", xmin)
      .add_option("xmax", "Maximum x coordinate", xmax)
      .add_option("ymin", "Minimum y coordinate", ymin)
      .add_option("ymax", "Maximum y coordinate", ymax)
      .add_option("filter", "Access only the queried region using the index", filter)
      .add_option("channel-list", "A list of channel IDs to draw", channelListStr)
      .add_option("color-list", "A list of colors for channels in hex code (#RRGGBB)", colorListStr);
    // Output
    pl.add_option("out", "Output image file", outFile, true)
      .add_option("verbose", "Verbose", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (filter && indexFile.empty())
        error("Index file must be provided when --filter is set");

    if (!checkOutputWritable(outFile))
        error("Output file is not writable: %s", outFile.c_str());

    if (!rangeFile.empty()) {
       readCoordRange(rangeFile, xmin, xmax, ymin, ymax);
    }
    if (xmin >= xmax || ymin >= ymax)
        error("Invalid range: xmin >= xmax or ymin >= ymax");

    bool selected = !channelListStr.empty();
    if (!selected && colorFile.empty())
        error("Either --in-color or both --channel-list and --color-list must be provided");

    std::vector<std::vector<int>> cmtx;
    std::unordered_map<int,std::vector<int32_t>> selectedMap;
    if (selected) { // parse selected channels
        if (colorListStr.empty())
            colorListStr = std::vector<std::string>{"144A74", "FF9900", "DD65E6", "FFEC11"}; // default colors
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
            if (line.empty()) continue;
            std::istringstream iss(line);
            int r,g,b;
            if (iss>>r>>g>>b) cmtx.push_back({r,g,b});
        }
    }

    // set up reader
    PixelResultReader reader(dataFile, indexFile, headerFile);
    int32_t k = reader.getK();
    if (k<=0) error("No factor columns found in header");
    if (filter) {
        int32_t ntiles = reader.query(xmin, xmax, ymin, ymax);
        if (ntiles <= 0)
            error("No data in the queried region");
        notice("Found %d tiles intersecting the queried region", ntiles);
    }

    if (scale<=0) error("--scale must be >0");

    // image dims
    int width  = int(std::floor((xmax-xmin)/scale))+1;
    int height = int(std::floor((ymax-ymin)/scale))+1;
    notice("Image size: %d x %d", width, height);
    if (width<=1||height<=1)
        error("Image dimensions are zero; check your bounds/scale");

    // accumulators
    cv::Mat3f sumImg(height, width, cv::Vec3f(0,0,0));
    cv::Mat1b countImg(height, width, uchar(0));
    // read & accumulate
    PixelFactorResult rec;
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
        if (xpix<0||xpix>=width||ypix<0||ypix>=height) continue;
        if (countImg(ypix, xpix)>=255) { nskip++; continue; }

        float R=0,G=0,B=0;
        bool valid=false;
        for (int i=0;i<k;++i) {
            int ch = rec.ks[i];
            double p = rec.ps[i];
            if (selected) {
                auto it = selectedMap.find(ch);
                if (it == selectedMap.end()) continue;
                auto& c = it->second;
                R += c[0]*p;
                G += c[1]*p;
                B += c[2]*p;
                valid = true;
            } else {
                if (ch<0 || ch>= (int)cmtx.size()) { valid=false; break; }
                R += cmtx[ch][0]*p;
                G += cmtx[ch][1]*p;
                B += cmtx[ch][2]*p;
                valid = true;
            }
        }
        if (!valid) continue;

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
