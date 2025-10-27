// Minibatch SLDA processor
#pragma once

#include "tiles2minibatch.hpp"

template<typename T>
class Tiles2SLDA : public Tiles2MinibatchBase {

public:
    Tiles2SLDA(int nThreads, double r,
        const std::string& _outPref, const std::string& _tmpDir,
        LatentDirichletAllocation& _lda,
        TileReader& tileReader, lineParserUnival& lineParser,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed = std::random_device{}(),
        double c = 20, double h = 0.7, double res = 1,
        int32_t M = 0, int32_t N = 0, int32_t k = 3,
        int32_t verbose = 0, int32_t debug = 0);

    void setFeatureNames(const std::vector<std::string>& names) {
        assert(names.size() == M_);
        featureNames = names;
    }
    void setLloydIter(int32_t nIter) { nLloydIter = nIter; }
    void setOutputCoordDigits(int32_t digits) { floatCoordDigits = digits; }
    void setOutputProbDigits(int32_t digits) { probDigits = digits; }
    void setOutputOptions(bool includeOrg, bool useTicket) {
        outputOriginalData = includeOrg;
        useTicketSystem = useTicket;
    }

    void run() override;

protected:
    using vec2f_t = Tiles2MinibatchBase::vec2f_t;

    int32_t debug_;
    int32_t M_, K_;
    bool weighted;
    size_t recordSize_ = 0;
    std::vector<FieldDef> schema_;
    LatentDirichletAllocation& lda;
    OnlineSLDA slda;
    lineParserUnival& lineParser;
    HexGrid hexGrid;
    int32_t nMoves;
    double anchorMinCount, distNu, distR;
    float pixelResolution;
    double eps_;
    int32_t nLloydIter = 1;
    MatrixXf pseudobulk; // K x M
    std::mutex pseudobulkMutex; // Protects pseudobulk

    int32_t initAnchorsHybrid(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors = nullptr);
    int32_t initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);
    int32_t makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);

    void processTile(TileData<T> &tileData, int threadId=0, int ticket = 0, vec2f_t* anchorPtr = nullptr);

    void writePseudobulkToTsv();
    void setExtendedSchema();

    void tileWorker(int threadId) override;
    void boundaryWorker(int threadId) override;
};
