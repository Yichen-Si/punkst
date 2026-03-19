// Minibatch SLDA processor
#pragma once

#include "numerical_utils.hpp"
#include "tiles2minibatch.hpp"
#include "lda.hpp"
#include "slda.hpp"

template<typename T>
class Tiles2SLDA : public Tiles2MinibatchBase<T> {

public:
    Tiles2SLDA(int nThreads, double supportRadius, double paddingRadius,
        const std::string& outPref, const std::string& tmpDir,
        LatentDirichletAllocation& lda,
        TileReader& tileReader, lineParserUnival& lineParser,
        const MinibatchIoConfig& ioConfig,
        unsigned int seed = std::random_device{}(),
        double c = 20, double weightAtAnchorDist = 0.3, double res = 1,
        int32_t N = 0, int32_t k = 3,
        int32_t verbose = 0, int32_t debug = 0,
        double hexSize = 1.0, int32_t nMoves = 1);

    void setLloydIter(int32_t nIter) { nLloydIter_ = nIter; }
    void set_background_prior(VectorXf& eta0, double a0, double b0, bool outputExpand = false);
    void set_background_prior(std::string& bgModelFile, double a0, double b0, bool outputExpand = false);
    int32_t getFactorCount() const override { return K_; }

    protected:
    using Base = Tiles2MinibatchBase<T>;
    using Base::debug_;
    using Base::lineParserPtr;
    using Base::useExtended_;
    using Base::topk_;
    using Base::M_;
    using Base::featureNames;
    using Base::probDigits;
    using Base::outPref;
    using Base::resultQueue;
    using Base::anchorQueue;
    using Base::pixelResolution_;
    using typename Base::ResultBuf;
    using vec2f_t = typename Base::vec2f_t;

    int32_t K_;
    LatentDirichletAllocation& lda_;
    OnlineSLDA slda_;
    lineParserUnival& lineParser_;
    HexGrid hexGrid_;
    const BCCGrid bccGrid_;
    int32_t nMoves_ = 1;
    double anchorMinCount_, supportRadius_, weightAtAnchorDist_;
    double eps_;
    int32_t nLloydIter_ = 1;
    MatrixXf pseudobulk_; // M x K
    RowMajorMatrixXf confusion_; // K x K
    std::mutex pseudobulkMutex_; // Protects pseudobulk
    bool fitBackground_ = false;

    int32_t initAnchorsHybrid(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors = nullptr);
    int32_t initAnchors(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, Minibatch& minibatch);
    double makeMinibatch(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, Minibatch& minibatch);
    double referenceAnchorDistance() const;

    void processTile(TileData<T> &tileData, int threadId, int ticket, vec2f_t* anchorPtr) override;
    void postRun() override;
    void onWorkerStart(int threadId) override;

    void writeGlobalMatrixToTsv();

};
