#pragma once

#include "numerical_utils.hpp"
#include "tiles2minibatch.hpp"
#include "hexgrid.h"
#include "tilereader.hpp"
#include "poisreg.hpp"
#include <limits>

using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::ArrayXf;
using Eigen::VectorXf;
using Eigen::SparseMatrix;
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct MultinomialLogReg {
    int M = 0;
    int C = 0;
    RowMajorMatrixXf W; // M x C
    Eigen::RowVectorXf b; // 1 x C

    MultinomialLogReg() {}
    MultinomialLogReg(const std::string& path) {
        load(path);
    }

    bool load(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) return false;

        fin.read(reinterpret_cast<char*>(&M), sizeof(int));
        fin.read(reinterpret_cast<char*>(&C), sizeof(int));
        W.resize(M, C);
        b.resize(C);
        fin.read(reinterpret_cast<char*>(W.data()), sizeof(float) * M * C);
        fin.read(reinterpret_cast<char*>(b.data()), sizeof(float) * C);
        fin.close();

        notice("%s: Loaded model with M=%d, C=%d", __func__, M, C);
        return true;
    }

    // Return n x C
    template <int StorageOrder>
    RowMajorMatrixXf predict(Eigen::SparseMatrix<float, StorageOrder>& mtx) const {
        RowMajorMatrixXf logits = mtx * W;
        logits.rowwise() += b;
        rowSoftmaxInPlace(logits);
        return logits;
    }

    // Return one row vector of size C
    template <typename SparseRowLike>
    Eigen::RowVectorXf predictOne(const SparseRowLike& vec) const {
        Eigen::RowVectorXf logits = b;
        logits += vec.transpose() * W;
        softmaxInPlace(logits);
        return logits;
    }

};


class PixelEM {
public:
    int debug_;
    MLEOptions mle_opts_;
    double size_factor_ = 1000.;

    struct EMstats {
        int32_t niter = 0;
        double  last_change = 0.;
        double  last_avg_internal_niters = 1;
    };

    PixelEM(int _debug = 0) : debug_(_debug) {}

    int32_t get_K() const { return K_; }
    int32_t get_M() const { return M_; }

    void set_em_options(int32_t max_iter, double tol = 1e-4, double  min_ct = 5) {
        max_iter_ = max_iter;
        tol_ = tol;
        min_ct_ = min_ct;
    }

    void run_em(Minibatch& batch, EMstats& stats, int max_iter = -1, double tol = -1) const {
        if (mlr_initialized_) {
            run_em_mlr(batch, stats, max_iter, tol);
            return;
        }
        if (!pnmf_initialized_) {
            error("%s: model not initialized, call init_pnmf or init_mlr first", __func__);
        }
        run_em_pnmf(batch, stats, max_iter, tol);
    }

    template<typename Derived>
    void init_mlr(const std::string& mlr_path, const Eigen::MatrixBase<Derived>& beta) {
        Eigen::RowVectorXd colSums = beta.colwise().sum();
        logBeta_ = (beta.array().rowwise() / colSums.array() + 1e-10).log().template cast<float>();
        M_ = logBeta_.rows();
        K_ = logBeta_.cols();
        if (!mlr_.load(mlr_path)) {
            error("%s: failed to load multinomial logistic regression model from %s", __func__, mlr_path.c_str());
        }
        if (M_ != mlr_.M || K_ != mlr_.C) {
            error("%s: dimension mismatch between %s and matrix", __func__, mlr_path.c_str());
        }
        mlr_initialized_ = true;
    }

    void run_em_mlr(Minibatch& batch, EMstats& stats, int max_iter = -1, double tol = -1) const;

    template<typename Derived>
    void init_pnmf(const Eigen::MatrixBase<Derived>& beta, MLEOptions opts, double L = 1000, bool exact = false) {
        beta_ = beta.template cast<double>();
        beta_f_ = beta.template cast<float>();
        Eigen::RowVectorXf colSums = beta_f_.colwise().sum();
        logBeta_ = (beta_f_.array().rowwise() / colSums.array() + 1e-10).log();
        M_ = beta_.rows();
        K_ = beta_.cols();
        if (M_ < 1 || K_ < 1) {
            error("%s: Invalid dimensions of beta", __func__);
        }
        mle_opts_ = opts;
        size_factor_ = L;
        exact_ = exact;
        pnmf_initialized_ = true;
    }

    void run_em_pnmf(Minibatch& batch, EMstats& stats, int max_iter = -1, double tol = -1) const;

    bool is_mlr_initialized() const {return mlr_initialized_;}
    bool is_pnmf_initialized() const {return pnmf_initialized_;}

private:
    int32_t K_ = 0, M_ = 0;
    MultinomialLogReg mlr_;
    RowMajorMatrixXd beta_;    // M x K
    RowMajorMatrixXf beta_f_;
    RowMajorMatrixXf logBeta_; // M x K
    bool    exact_       = false;
    int32_t max_iter_    = 20;
    double  tol_         = 1e-4;
    double  min_ct_      = 5;
    bool    pnmf_initialized_ = false;
    bool    mlr_initialized_ = false;
};

template<typename T>
class Tiles2NMF : public Tiles2MinibatchBase<T> {
public:
    Tiles2NMF(int nThreads, double r,
               const std::string& outPref, const std::string& tmpDir,
               const PixelEM& empois, TileReader& tileReader,
               lineParserUnival& lineParser, HexGrid& hexGrid, int32_t nMoves,
               unsigned int seed = std::random_device{}(),
               double c = 20.0, double h = 0.7, double res = 1.0, int32_t topk = 3,
               int32_t verbose = 0, int32_t debug = 0);

protected:
    using Base = Tiles2MinibatchBase<T>;
    using Base::debug_;
    using Base::lineParserPtr;
    using Base::useExtended_;
    using Base::topk_;
    using Base::M_;
    using Base::featureNames;
    using Base::resultQueue;
    using Base::outputOriginalData_;
    using Base::outputAnchor_;
    using Base::anchorQueue;
    using typename Base::ProcessedResult;
    using vec2f_t = typename Base::vec2f_t;

    int32_t K_;
    const PixelEM&  empois_;
    lineParserUnival& lineParser_;
    HexGrid&          hexGrid_;
    int32_t           nMoves_;
    unsigned int      seed_;
    double            anchorMinCount_;
    double            distNu_, distR_;
    float             pixelResolution_;
    int32_t           verbose_;

    // MatrixXf pseudobulk_;
    // std::mutex pseudobulkMutex_;

    int32_t initAnchors(TileData<T>& tileData,
                        std::vector<cv::Point2f>& anchors,
                        Minibatch& minibatch);
    int32_t makeMinibatch(TileData<T>& tileData,
                          std::vector<cv::Point2f>& anchors,
                          Minibatch& minibatch);
    void processTile(TileData<T>& tileData,
                     int threadId, int ticket, vec2f_t* anchorPtr) override;
    // void writePseudobulkToTsv();
};
