#include "topic_svb.hpp"

class ConditionalDEtest : public LDA4Hex {
public:
    ConditionalDEtest(int32_t nFold, int32_t nContrast,
        int32_t nThreads, int32_t seed,
        const std::string& dataFile, const std::string& metaFile,
        const std::string& labelFile,
        const std::string& modelFile,
        int32_t maxIter = 100, double mDelta = -1., int32_t bSize = 1024,
        int32_t debug = 0, int32_t verbose = 0) :
        LDA4Hex(metaFile, 0),
        nFold_(nFold), nContrast_(nContrast), nThreads_(nThreads), seed_(seed),
        dataFile_(dataFile), labelFile_(labelFile),
        maxIter_(maxIter), mDelta_(mDelta), debug_(debug), verbose_(verbose)
    {
        initialize(0, priorMatrix_, modelFile, 1.0);
        beta_ = priorMatrix_;
        for (int32_t i = 0; i < beta_.rows(); ++i) {
            double row_sum = beta_.row(i).sum();
            if (row_sum > 0) {
                beta_.row(i) /= row_sum;
            }
        }
        batchSize = bSize;
        if (seed_ <= 0) {
            seed_ = std::random_device{}();
        }
        rng_.seed(seed_);
        modelList_.resize(nFold_);
        reset();
    }

    void reset() {
        dataStream_.open(dataFile_);
        labelStream_.open(labelFile_);
        fileopen = dataStream_.is_open() && labelStream_.is_open();
        if (!fileopen) {
            error("%s: Failed to open input file(s)", __func__);
        }
        makeSplits();
        for (int32_t k = 0; k < nFold_; ++k) {
            int32_t M = static_cast<int32_t>(trainIdx_[k].size());
            MatrixXd modelMatrix = priorMatrix_(Eigen::placeholders::all, Eigen::ArrayXi::Map(trainIdx_[k].data(), M));
            modelList_[k].reset();
            modelList_[k] = std::make_unique<LatentDirichletAllocation>(
                                modelMatrix, seed_, nThreads_);
            modelList_[k]->set_svb_parameters(maxIter_, mDelta_);
        }
        l_sum.clear(); x_sum.clear(); yx.clear(); yl.clear();
        l_sum.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        x_sum.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        yx.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        yl.resize(nContrast_, Eigen::VectorXd::Zero(M_));
    }

    std::vector<Eigen::VectorXd> processAll() {
        int32_t v = 0, ntot = 0;
        while (fileopen) {
            int32_t n = processMinibatch();
            if (n >= 0) {
                ntot += n;
                if (ntot > v * verbose_) {
                    notice("Processed %d documents", ntot);
                    v++;
                }
            }
            if (debug_ && ntot > debug_) {
                break;
            }
        }
        if (dataStream_.is_open()) {
            dataStream_.close();
        }
        if (labelStream_.is_open()) {
            labelStream_.close();
        }
        return computeScores();
    }

    int32_t processMinibatch() {
        if (!fileopen) {
            return -1;
        }
        fileopen = readMinibatch(dataStream_);
        int32_t nDocs = static_cast<int32_t>(minibatch.size());
        if (nDocs == 0) {return 0;}
        int32_t nLabels = readMinibatchLabels(nDocs);
        if (nLabels != nDocs) {
            error("%s: Mismatch between number of documents and labels", __func__);
        }
        for (int r = 0; r < nContrast_; ++r) {
            computeOneContrast(r);
        }
        return nDocs;
    }

    std::vector<Eigen::VectorXd> computeScores() {
        std::vector<Eigen::VectorXd> scores(nContrast_);
        for (int r = 0; r < nContrast_; ++r) {
            VectorXd I = yl[r].array() * x_sum[r].array() / (l_sum[r].array() + 0.5);
            VectorXd U = yx[r].array() - I.array();
if (debug_ % 2 == 1) {
    std::cout << r << "l_sum:\n  ";
    std::cout << l_sum[r].transpose() << "\n";
    std::cout << r << "x_sum:\n  ";
    std::cout << x_sum[r].transpose() << "\n";
    std::cout << r << " I:\n  ";
    std::cout << I.transpose() << "\n";
    std::cout << r << " U:\n  ";
    std::cout << U.transpose() << "\n";
}
            scores[r] = U.array().pow(2) / I.array();
        }
        return scores;
    }

    const std::vector<std::vector<int32_t>>& getTrainIndices() const {
        return trainIdx_;
    }
    const std::vector<std::vector<int32_t>>& getHeldoutIndices() const {
        return heldoutIdx_;
    }

private:

    int32_t nThreads_;
    int32_t nFold_;
    int32_t nContrast_;
    int32_t maxIter_;
    double mDelta_;
    MatrixXd priorMatrix_, beta_; // K x M
    int32_t seed_, debug_, verbose_;
    std::mt19937 rng_;

    std::string dataFile_, labelFile_;
    std::ifstream labelStream_, dataStream_;
    bool fileopen;

    std::vector<std::unique_ptr<LatentDirichletAllocation> > modelList_;
    std::vector<std::vector<int32_t>> featureIdxMaps_;
    std::vector<std::vector<int32_t>> trainIdx_, heldoutIdx_;

    std::vector<Eigen::VectorXi> masks_, contrasts_;
    std::vector<Eigen::VectorXd> l_sum, x_sum, yx, yl;

    void makeSplits() {
        assert(M_  > 0);
        assert(nFold_ >= 2 && nFold_ <= M_);

        std::vector<int32_t> perm(M_);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        const int32_t base = M_ / nFold_;  // ⌊M/k⌋
        const int32_t extra = M_ % nFold_; // first ‘extra’ folds get +1
        trainIdx_.clear(); trainIdx_.resize(nFold_);
        heldoutIdx_.clear(); heldoutIdx_.resize(nFold_);
        featureIdxMaps_.clear();
        featureIdxMaps_.resize(nFold_, std::vector<int32_t>(M_, -1));

        auto it = perm.begin();
        for (int32_t k = 0; k < nFold_; ++k) {
            int32_t foldSize = base + (k < extra ? 1 : 0);
            heldoutIdx_[k].assign(it, it + foldSize);
            trainIdx_[k].reserve(M_ - foldSize);
            trainIdx_[k].insert(trainIdx_[k].end(), perm.begin(), it);
            trainIdx_[k].insert(trainIdx_[k].end(), it + foldSize, perm.end());
            for (std::size_t j = 0; j < trainIdx_[k].size(); ++j) {
                featureIdxMaps_[k][trainIdx_[k][j]] = static_cast<int32_t>(j);
            }
            it += foldSize;
        }
    }

    int32_t readMinibatchLabels(int32_t n) {
        std::string line;
        int32_t nlocal = 0;
        masks_.resize(nContrast_);
        contrasts_.resize(nContrast_);
        for (int i = 0; i < nContrast_; ++i) {
            masks_[i] = Eigen::VectorXi::Zero(n);
            contrasts_[i] = Eigen::VectorXi::Zero(n);
        }
        while (nlocal < n) {
            if (!std::getline(labelStream_, line)) {
                return nlocal;
            }
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < nContrast_) {
                error("%s: incomplete label line: %s", __func__, line.c_str());
            }
            std::vector<int32_t> labels(nContrast_);
            for (int32_t i = 0; i < nContrast_; ++i) {
                if (!str2num<int32_t>(tokens[i], labels[i]) || labels[i] > 1) {
                    error("%s: invalid label value '%s' in line: %s", __func__, tokens[i].c_str(), line.c_str());
                }
                masks_[i][nlocal] = labels[i] == 0 | labels[i] == 1;
                contrasts_[i][nlocal] = labels[i] == 1;
            }
            ++nlocal;
        }
        return nlocal;
    }

    void computeOneContrast(int32_t r) {
        int32_t nDocs = static_cast<int32_t>(minibatch.size());
        if (nDocs == 0) {return;}
        std::vector<Document> docs; docs.reserve(nDocs);
        std::vector<double> xsum; xsum.reserve(nDocs);
        std::vector<double> yvec; yvec.reserve(nDocs);
        for (int i = 0; i < nDocs; ++i) {
            if (!masks_[r][i]) {
                continue;
            }
            const auto& doc = minibatch[i];
            size_t n = doc.ids.size();
            if (masks_[r][i]) {
                for (size_t j = 0; j < n; ++j) {
                    x_sum[r][doc.ids[j]] += doc.cnts[j];
                }
            }
            if (contrasts_[r][i]) {
                for (size_t j = 0; j < n; ++j) {
                    yx[r][doc.ids[j]] += doc.cnts[j];
                }
            }
            xsum.push_back(std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0));
            yvec.push_back((double) contrasts_[r][i]);
            Document newDoc;
            for (size_t j = 0; j < n; ++j) {
                int32_t idx = featureIdxMaps_[r][doc.ids[j]];
                if (idx < 0) {
                    continue;
                }
                newDoc.ids.push_back(idx);
                newDoc.cnts.push_back(doc.cnts[j]);
            }
            docs.push_back(std::move(newDoc));
        }
        MatrixXd doc_topic = modelList_[r]->transform(docs); // n x K
        for (int i = 0; i < doc_topic.rows(); ++i) {
            double sum = doc_topic.row(i).sum();
            if (sum > 0) doc_topic.row(i) /= sum;
        }
        auto colIdx = Eigen::ArrayXi::Map(heldoutIdx_[r].data(), heldoutIdx_[r].size());
        VectorXd s = Eigen::Map<VectorXd>(xsum.data(), xsum.size());
        Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(yvec.data(), yvec.size());
        MatrixXd lambda = doc_topic * beta_(Eigen::placeholders::all, colIdx);
        lambda.array().colwise() *= s.array();
        l_sum[r](colIdx) += lambda.colwise().sum();
        yl[r](colIdx) += lambda.transpose() * y;

if (debug_ % 2 == 1) {
    VectorXd max_lambda = lambda.rowwise().maxCoeff();
    std::cout << r << " lambda * s:\n  ";
    for (int i = 0; i < 10; ++i) {
        std::cout << i << " " << max_lambda(i) << ", ";
    }
    std::cout << "\n";
    std::cout << r << " l_sum:\n  ";
    for (int i = 0; i < 10; ++i) {
        std::cout << l_sum[r](heldoutIdx_[r][i]) << ", ";
    }
    std::cout << "\n";
    std::cout << r << " yl:\n  ";
    for (int i = 0; i < 10; ++i) {
        std::cout << yl[r](heldoutIdx_[r][i]) << ", ";
    }
    std::cout << "\n";
}

    }

};
