#include "topic_svb.hpp"

class GlobalDEtest : public LDA4Hex {
public:
    GlobalDEtest(int32_t nFold, int32_t nContrast,
        int32_t nThreads, int32_t seed,
        const std::string& dataFile, HexReader& reader,
        const std::string& labelFile,
        const std::string& modelFile,
        int32_t maxIter = 100, double mDelta = -1., int32_t bSize = 1024,
        int32_t debug = 0, int32_t verbose = 0) :
        LDA4Hex(reader, 0),
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
    }

    std::vector<Eigen::VectorXd> globalRun() {
        globalReset();
        int32_t v = 0, ntot = 0;
        while (fileopen) {
            int32_t n = globalProcessMinibatch();
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
        closeStreams();
        return globalScores();
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
    bool featureSplit_;

    std::string dataFile_, labelFile_;
    std::ifstream labelStream_, dataStream_;
    bool fileopen;

    std::vector<std::unique_ptr<LatentDirichletAllocation> > modelList_;
    std::vector<std::vector<int32_t>> featureIdxMaps_;
    std::vector<std::vector<int32_t>> trainIdx_, heldoutIdx_;
    std::vector<std::vector<bool>> masks_, contrasts_;

    // Sufficient statistics for global conditional test
    // x_sum[c][j]: \sum x_ij (over all docs used in contrast c)
    // l_sum[c][j]: expected x_sum[c][j]
    // yx[c][j]: \sum y_i x_ij (over docs with label 1 in contrast c)
    // yl[c][j]: extected yx[c][j]
    std::vector<Eigen::VectorXd> x_sum, yx;
    std::vector<Eigen::VectorXd> l_sum, yl;

    // Randomly split features
    void makeSplits() {
        assert(M_  > 0);
        assert(nFold_ >= 2 && nFold_ <= M_);

        std::vector<int32_t> perm(M_);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        const int32_t base = M_ / nFold_;
        const int32_t extra = M_ % nFold_;
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

    // Read contrast labels for the next n data points
    int32_t readMinibatchLabels(int32_t n) {
        std::string line;
        int32_t nlocal = 0;
        masks_.resize(nContrast_);
        contrasts_.resize(nContrast_);
        for (int i = 0; i < nContrast_; ++i) {
            masks_[i].assign(n, false);
            contrasts_[i].assign(n, false);
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
            for (int32_t i = 0; i < nContrast_; ++i) {
                int32_t label;
                if (!str2num<int32_t>(tokens[i], label) || label > 1) {
                    error("%s: invalid label value '%s' in line: %s", __func__, tokens[i].c_str(), line.c_str());
                }
                masks_[i][nlocal] = label == 0 | label == 1;
                contrasts_[i][nlocal] = label == 1;
            }
            ++nlocal;
        }
        return nlocal;
    }

    int32_t globalProcessMinibatch() {
        if (!fileopen) {
            return -1;
        }
        fileopen = readMinibatch(dataStream_);
        int32_t nDocs = static_cast<int32_t>(minibatch.size());
        if (nDocs == 0) {return 0;}
        int32_t nLabels = readMinibatchLabels(nDocs); // unsafe, rely on the files being aligned
        if (nLabels != nDocs) {
            error("%s: Mismatch between number of documents and labels", __func__);
        }
        for (int c = 0; c < nContrast_; ++c) {
            globalOneContrast(c);
        }
        return nDocs;
    }

    // Compute score test statistics
    std::vector<Eigen::VectorXd> globalScores() {
        std::vector<Eigen::VectorXd> scores(nContrast_);
        for (int c = 0; c < nContrast_; ++c) {
            VectorXd I = yl[c].array() * (x_sum[c].array() / l_sum[c].array().max(0.5));
            VectorXd U = yx[c].array() - I.array();
            scores[c] = U.array().pow(2) / I.array();
        }
        return scores;
    }

    // Given one minibatch, update sufficient statistics for one contrast
    void globalOneContrast(int32_t c) {
        int32_t nDocs = static_cast<int32_t>(minibatch.size());
        if (nDocs == 0) {return;}

        std::vector<double> xsum; xsum.reserve(nDocs);
        std::vector<double> yvec; yvec.reserve(nDocs);
        for (int i = 0; i < nDocs; ++i) {
            if (!masks_[c][i]) {
                continue;
            }
            const auto& doc = minibatch[i];
            size_t n = doc.ids.size();
            for (size_t j = 0; j < n; ++j) {
                x_sum[c][doc.ids[j]] += doc.cnts[j];
            }
            if (contrasts_[c][i]) {
                for (size_t j = 0; j < n; ++j) {
                    yx[c][doc.ids[j]] += doc.cnts[j];
                }
            }
            xsum.push_back(std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0));
            yvec.push_back((double) (int) contrasts_[c][i]);
        }
        VectorXd s = Eigen::Map<VectorXd>(xsum.data(), xsum.size());
        VectorXd y = Eigen::Map<VectorXd>(yvec.data(), yvec.size());

        // Using all features
        if (nFold_ <= 1) {
            MatrixXd doc_topic = lda->transform(minibatch); // n x K
            for (int i = 0; i < doc_topic.rows(); ++i) {
                double sum = doc_topic.row(i).sum();
                if (sum > 0) doc_topic.row(i) /= sum;
            }
            // compute expected counts
            MatrixXd lambda = doc_topic * beta_;
            lambda.array().colwise() *= s.array(); // n_i * <\theta_i, \beta_j>
            l_sum[c] += lambda.colwise().sum();
            yl[c] += lambda.transpose() * y;
            return;
        }

        // Using feature splits
        for (int t = 0; t < nFold_; ++t) {
            std::vector<Document> docs; docs.reserve(nDocs);
            for (int i = 0; i < nDocs; ++i) {
                if (!masks_[c][i]) {
                    continue;
                }
                const auto& doc = minibatch[i];
                size_t n = doc.ids.size();
                Document newDoc;
                for (size_t j = 0; j < n; ++j) { // not very efficient
                    int32_t idx = featureIdxMaps_[t][doc.ids[j]];
                    if (idx < 0) {
                        continue;
                    }
                    newDoc.ids.push_back(idx);
                    newDoc.cnts.push_back(doc.cnts[j]);
                }
                docs.push_back(std::move(newDoc));
            }
            MatrixXd doc_topic = modelList_[t]->transform(docs); // n x K
            for (int i = 0; i < doc_topic.rows(); ++i) {
                double sum = doc_topic.row(i).sum();
                if (sum > 0) doc_topic.row(i) /= sum;
            }
            auto colIdx = Eigen::ArrayXi::Map(heldoutIdx_[t].data(), heldoutIdx_[t].size());
            // compute expected counts
            MatrixXd lambda = doc_topic * beta_(Eigen::placeholders::all, colIdx);
            lambda.array().colwise() *= s.array(); // n_i * <\theta_i, \beta_j>
            l_sum[c](colIdx) += lambda.colwise().sum();
            yl[c](colIdx) += lambda.transpose() * y;
        }
    }

    void globalReset() {
        resetStreams();
        x_sum.clear(); yx.clear();
        x_sum.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        yx.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        l_sum.clear(); yl.clear();
        l_sum.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        yl.resize(nContrast_, Eigen::VectorXd::Zero(M_));
        if (nFold_ <= 1) {return;}

        makeSplits();
        modelList_.resize(nFold_);
        for (int32_t k = 0; k < nFold_; ++k) {
            int32_t M = static_cast<int32_t>(trainIdx_[k].size());
            RowMajorMatrixXd modelMatrix = priorMatrix_(Eigen::placeholders::all, Eigen::ArrayXi::Map(trainIdx_[k].data(), M)); // subset to training features
            modelList_[k].reset();
            modelList_[k] = std::make_unique<LatentDirichletAllocation>(
                                modelMatrix, seed_, nThreads_);
            modelList_[k]->set_svb_parameters(maxIter_, mDelta_);
        }
    }

    void resetStreams() {
        dataStream_.open(dataFile_);
        labelStream_.open(labelFile_);
        fileopen = dataStream_.is_open() && labelStream_.is_open();
        if (!fileopen) {
            error("%s: Failed to open input file(s)", __func__);
        }
        // skip headers in labelStream
        std::string line;
        uint64_t pos = 0;
        while (std::getline(labelStream_, line)) {
            if (line.empty() && line[0] != '#') {
                labelStream_.seekg(pos);
                break;
            }
            pos = labelStream_.tellg();
        }
    }
    void closeStreams() {
        if (dataStream_.is_open()) {
            dataStream_.close();
        }
        if (labelStream_.is_open()) {
            labelStream_.close();
        }
    }

};
