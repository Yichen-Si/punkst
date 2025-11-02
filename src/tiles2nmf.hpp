// Minibatch Poisson regression (log1p link) processor.
#pragma once

#include "tiles2minibatch.hpp"
#include "pixdecode.hpp"
#include "poisnmf.hpp"
#include "hexgrid.h"
#include "tilereader.hpp"

#include <random>
#include <limits>
#include <cassert>

template<typename T>
class Tiles2NMF : public Tiles2MinibatchBase {
public:
    Tiles2NMF(int nThreads, double r,
               const std::string& outPref, const std::string& tmpDir,
               PoissonLog1pNMF& nmf, EMPoisReg& empois,
               TileReader& tileReader, lineParserUnival& lineParser,
               HexGrid& hexGrid, int32_t nMoves,
               unsigned int seed = std::random_device{}(),
               double c = 20.0, double h = 0.7, double res = 1.0, int32_t topk = 3,
               int32_t verbose = 0, int32_t debug = 0);

    void set_em_options(int32_t max_iter, double tol) {
        max_iter_ = max_iter;
        tol_ = tol;
    }

    void run() override;

protected:
    using vec2f_t = Tiles2MinibatchBase::vec2f_t;

    PoissonLog1pNMF&  nmf_;
    EMPoisReg&        empois_;
    lineParserUnival& lineParser_;
    HexGrid           hexGrid_;
    int32_t           nMoves_;
    unsigned int      seed_;
    double            anchorMinCount_;
    double            distNu_, distR_;
    float             pixelResolution_;
    int32_t           debug_;
    int32_t           verbose_;
    int32_t           max_iter_ = 20;
    double            tol_ = 1e-4;

    int32_t K_;
    double  eps_ = std::numeric_limits<double>::epsilon();

    // MatrixXf pseudobulk_;
    // std::mutex pseudobulkMutex_;

    int32_t initAnchors(TileData<T>& tileData,
                        std::vector<cv::Point2f>& anchors,
                        Minibatch& minibatch);
    int32_t makeMinibatch(TileData<T>& tileData,
                          std::vector<cv::Point2f>& anchors,
                          Minibatch& minibatch);
    void processTile(TileData<T>& tileData,
                     int threadId = 0, int ticket = 0);

    // void writePseudobulkToTsv();

    void tileWorker(int threadId) override;
    void boundaryWorker(int threadId) override;
};
