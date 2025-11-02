// Minibatch SLDA processor
#pragma once

#include "tiles2minibatch.hpp"
#include "lda.hpp"
#include "pixdecode.hpp"

template<typename T>
class Tiles2SLDA : public Tiles2MinibatchBase {

public:
    Tiles2SLDA(int nThreads, double r,
        const std::string& outPref, const std::string& tmpDir,
        LatentDirichletAllocation& lda,
        TileReader& tileReader, lineParserUnival& lineParser,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed = std::random_device{}(),
        double c = 20, double h = 0.7, double res = 1,
        int32_t N = 0, int32_t k = 3,
        int32_t verbose = 0, int32_t debug = 0);

    void setLloydIter(int32_t nIter) { nLloydIter_ = nIter; }

    void run() override;

protected:
    using vec2f_t = Tiles2MinibatchBase::vec2f_t;

    int32_t debug_;
    int32_t K_;
    LatentDirichletAllocation& lda_;
    OnlineSLDA slda_;
    lineParserUnival& lineParser_;
    HexGrid& hexGrid_;
    int32_t nMoves_;
    double anchorMinCount_, distNu_, distR_;
    float pixelResolution_;
    double eps_;
    int32_t nLloydIter_ = 1;
    MatrixXf pseudobulk_; // K x M
    std::mutex pseudobulkMutex_; // Protects pseudobulk

    int32_t initAnchorsHybrid(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors = nullptr);
    int32_t initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);
    int32_t makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);

    void processTile(TileData<T> &tileData, int threadId=0, int ticket = 0, vec2f_t* anchorPtr = nullptr);

    void writePseudobulkToTsv();

    void tileWorker(int threadId) override;
    void boundaryWorker(int threadId) override;
};
