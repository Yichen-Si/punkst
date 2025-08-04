#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdint>
#include <set>
#include <unordered_set>
#include <optional>
#include "error.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

class MarkerSelector {
public:

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> betas; // M x K

    struct featureInfo {
        std::string name;
        uint64_t totalOcc;
        uint64_t totalNeighbor;
    };

    struct markerSetInfo {
        std::string name;
        uint64_t count;
        struct neighborInfo {
            std::string name;
            uint64_t count;
            double qij, qji;
            uint32_t rij, rji;
            neighborInfo(std::string name, uint64_t count, double qij, double qji, uint32_t rij, uint32_t rji) :
                name(name), count(count), qij(qij), qji(qji), rij(rij), rji(rji) {};
            neighborInfo() {};
        };
        std::vector<neighborInfo> neighbors;
        markerSetInfo() {};
        markerSetInfo(const std::string& name, uint64_t count) : name(name), count(count) {};
    };

    MarkerSelector(const std::string& featureFile, const std::string& mtxFile, bool binary, bool dense, int valueBytes, int minCount = 1, int32_t verbose = 0, std::vector<std::string>* whilteList = nullptr) : verbose_(verbose) {
        if (whilteList != nullptr) {
            whiteList_.insert(whilteList->begin(), whilteList->end());
        }
        loadGeneInfo(featureFile, minCount);
        if (dense) {
            loadDenseCooccurrenceMatrix(mtxFile, binary, valueBytes);
        } else {
            loadCooccurrenceMatrix(mtxFile, binary, valueBytes);
        }
    }

    // Select K anchors, optionally fixing a subset
    void selectMarkers(int K, std::vector<std::string>& selectedAnchors) {
        if (Q_.rows() < K) {
            error("Number of markers %d is larger than the number of features %d", K, Q_.rows());
        }
        anchors.clear();
        anchors.reserve(K);

        // Add valid fixed anchors
        std::unordered_set<int> fixedSet;
        for (auto &name : selectedAnchors) {
            auto it = nameToIdx_.find(name);
            if (it == nameToIdx_.end()) continue;
            int idx = it->second;
            if (validGene_[idx] && fixedSet.count(idx) == 0) {
                anchors.push_back(idx);
                fixedSet.insert(idx);
            }
        }
        int F = (int)anchors.size();
        if (F >= K) {
            selectedAnchors.resize(K);
             for (int i = 0; i < K; ++i)
                selectedAnchors[i] = features_[anchors[i]].name;
            warning("%s: >= %d (%d) fixed anchors are provided", __func__, K, F);
            return; // All anchors were fixed
        }
        notice("%s: received %d fixed anchors", __func__, F);

        // Build basis from the fixed anchors using QR
        Eigen::MatrixXd basis(M_, F);
        for(int k=0; k<F; ++k)
            basis.col(k) = Q_.row(anchors[k]);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
        Eigen::MatrixXd B = qr.householderQ() * Eigen::MatrixXd::Identity(M_, F);
        // Greedy
        for (int k = F; k < K; ++k) {
            double bestD2 = -1;
            int    bestIdx = -1;
            // Project onto the orthogonal complement of the current basis
            for (auto i : validIndex_) {
                if (fixedSet.count(i)) continue;
                Eigen::VectorXd r = Q_.row(i);
                double d2;
                if (k > 0) { // If there's a basis to project onto
                    Eigen::VectorXd temp_proj = B.transpose() * r;
                    Eigen::VectorXd proj = B * temp_proj;
                    d2 = (r - proj).squaredNorm();
                } else {
                    d2 = r.squaredNorm();
                }
                if (d2 > bestD2) { // find the largest residual
                    bestD2 = d2;
                    bestIdx = i;
                }
            }
            anchors.push_back(bestIdx);
            fixedSet.insert(bestIdx);

            // Update the basis with the new anchor
            Eigen::VectorXd new_vec = Q_.row(bestIdx);
            if (k > 0) {
                new_vec -= B * (B.transpose() * new_vec);
            }
            B.conservativeResize(Eigen::NoChange, k + 1);
            B.col(k) = new_vec.normalized();
            if (verbose_ > 0) {
                notice("Selected %d-th anchor: %s (%d)", k+1, features_[bestIdx].name.c_str(), features_[bestIdx].totalOcc);
            }
        }

        // Refine
        Eigen::MatrixXd A(M_, K);
        for (int i = 0; i < K; ++i)
            A.col(i) = Q_.row(anchors[i]);
        Eigen::HouseholderQR<Eigen::MatrixXd> full_qr(A);
        Eigen::MatrixXd Q_basis = full_qr.householderQ() * Eigen::MatrixXd::Identity(M_, K);
        for (int t = F; t < K; ++t) {
            fixedSet.clear();
            for (int j = 0; j < K; ++j) {
                if (j != t) fixedSet.insert(anchors[j]);
            }
            // Create a basis of the K-1 other vectors
            Eigen::MatrixXd B_others(M_, K - 1);
            int col_idx = 0;
            for(int j=0; j < K; ++j) {
                if (j != t) {
                    B_others.col(col_idx++) = Q_basis.col(j);
                }
            }
            double bestD2 = -1.0;
            int bestI = anchors[t];
            for (auto i : validIndex_) {
                if (fixedSet.count(i)) continue;
                Eigen::VectorXd qi = Q_.row(i);
                Eigen::VectorXd proj_others = B_others * (B_others.transpose() * qi);
                double d2 = (qi - proj_others).squaredNorm();
                if (d2 > bestD2) {
                    bestD2 = d2;
                    bestI = i;
                }
            }
            if (bestI != anchors[t]) {
                if (verbose_ > 0) {
                    notice("Replaced %d-th anchor %s with %s", t, features_[anchors[t]].name.c_str(), features_[bestI].name.c_str());
                }
                anchors[t] = bestI;
                // Update the basis for the next refinement iteration
                A.col(t) = Q_.row(bestI);
                full_qr.compute(A);
                Q_basis = full_qr.householderQ() * Eigen::MatrixXd::Identity(M_, K);
            }
        }

        selectedAnchors.resize(K);
        for (int i = 0; i < K; ++i) {
            selectedAnchors[i] = features_[anchors[i]].name;
        }
    }

    // Compute P(w=j|z=k) (or P(w=j,z=k))
    void computeTopicDistribution(int max_iter = 500, double tol = 1e-6, int threads = -1, bool weightByCounts = false) {
        if (anchors.empty()) {
            error("No anchors selected, please call selectMarkers(K, ...) first");
        }
        std::optional<tbb::global_control> gc;
        if (threads > 0) {
            gc.emplace(tbb::global_control::max_allowed_parallelism, threads);
        }
        const int M = M_;
        const int K = static_cast<int>(anchors.size());
        betas.resize(M, K);

        // Pre‐extract the K anchor rows
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Qs(K, M);
        for (int k = 0; k < K; ++k) {
            Qs.row(k) = Q_.row(anchors[k]);
        }

        // Marginal probabilities
        tbb::combinable<Eigen::VectorXd> pk_acc{
            [&]{ return Eigen::VectorXd::Zero(K); }
        };

        tbb::parallel_for(
            tbb::blocked_range<int>(0, M),
            [&](const tbb::blocked_range<int>& range) {

            Eigen::VectorXd c(K), c_new(K), g(K);
            Eigen::RowVectorXd r(M);
            auto& local_pk = pk_acc.local();

            for (int i = range.begin(); i < range.end(); ++i) {
                Eigen::RowVectorXd q = Q_.row(i);
                // Initialize weights uniformly
                c.setConstant(1.0 / K);
                // Iterative multiplicative‐updates
                int iter = 0;
                double diff = 0;
                for (; iter < max_iter; ++iter) {
                    // r_j = ∑_k c_k * Qs[k][j]
                    r = c.transpose() * Qs;
                    // g_k = ∑_j p_j * Qs[k][j] / r_j
                    g = Qs * (q.array() / r.array()).matrix().transpose();
                    // Update and renormalize
                    c_new = c.array() * g.array();
                    double norm = c_new.sum();
                    if (norm <= 0) break;
                    c_new /= norm;
                    // Convergence check
                    diff = (c_new - c).cwiseAbs().maxCoeff();
                    c.swap(c_new);
                    if (diff < tol) break;
                    if (verbose_ > 2 && iter % 100 == 0) {
                        notice("%s: %d-th feature %s, iter %d, max abs diff %.3e", __FUNCTION__, i, features_[i].name.c_str(), iter, diff);
                    }
                }
                if (verbose_ > 1 || (verbose_ > 0 && i % 1000 == 0)) {
                    notice("%s: solved %d-th feature in %d iterations, final max abs diff %.3e", __FUNCTION__, i, iter, diff);
                }
                // Store result
                betas.row(i) = c * rowWeights_(i);
                if (weightByCounts) {
                    local_pk = local_pk.array() + c.array() * features_[i].totalOcc;
                }
            }
        });
        // column normalize betas
        Eigen::RowVectorXd colsums = betas.colwise().sum();
        for (int k = 0; k < K; ++k) {
            if (colsums(k) > 1e-8) {
                betas.col(k) /= colsums(k);
            }
        }
        if (weightByCounts) {
            Eigen::RowVectorXd pk(K);
            pk = (pk_acc.combine([](auto &a, auto &b){ return a + b; })).transpose();
            betas.array().rowwise() *= pk.array();
        }
    }

    std::unordered_map<uint32_t, featureInfo> const & getFeatureInfo() const {
        return features_;
    }

    std::vector<markerSetInfo> findNeighborsToAnchors(int32_t m, int threads = -1, double maxRankFraction = -1) {
        std::optional<tbb::global_control> gc;
        if (threads > 0) {
            gc.emplace(tbb::global_control::max_allowed_parallelism, threads);
        }
        if (!rankMatrixBuilt) {
            buildRankMatrix();
        }
        uint32_t maxRank;
        if (maxRankFraction <= 0) {
            maxRank = M_;
        } else {
            maxRank = (uint32_t)(maxRankFraction * M_);
        }
        std::vector<markerSetInfo> result;
        for (int i : anchors) {
            // collect (feature, score)
            std::vector<std::pair<int,int>> cand;
            cand.reserve(M_);
            for (int j = 0; j < M_; ++j) {
                if (j == i) continue;
                int sc = std::max(ranks_(i, j), ranks_(j, i));
                if (sc > maxRank) continue;
                cand.emplace_back(j, sc);
            }
            // partial sort to m
            if ((int)cand.size() > m) {
                std::nth_element(
                    cand.begin(), cand.begin() + m, cand.end(),
                    [](auto &x, auto &y){ return x.second < y.second; }
                );
                cand.resize(m);
            }
            // final sort of the top-m by score
            std::sort(
                cand.begin(), cand.end(),
                [](auto &x, auto &y){ return x.second < y.second; }
            );

            markerSetInfo res(features_[i].name, features_[i].totalOcc);
            uint32_t kept = 0;
            for (auto &p : cand) {
                int j = p.first;
                if (Q_(i, j) == 0) continue;
                res.neighbors.emplace_back(
                    features_[j].name, features_[j].totalOcc,
                    Q_(i, j), Q_(j, i), ranks_(i, j), ranks_(j, i)
                );
                kept++;
            }
            if (verbose_ > 0) {
                std::stringstream ss;
                for (uint32_t j = 0; j < kept; ++j) {
                    auto &p = cand[j];
                    ss << features_[p.first].name << "(" << p.second << ", " << features_[p.first].totalOcc << ") ";
                }
                notice("Neighbors of %s: %s", features_[i].name.c_str(), ss.str().c_str());
            }
            result.push_back(std::move(res));
        }
        return result;
    }

private:

    uint32_t M_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q_;
    Eigen::VectorXd rowWeights_;
    std::unordered_map<uint32_t, featureInfo> features_;
    std::vector<uint32_t> validIndex_;
    std::vector<bool> validGene_;
    std::unordered_map<std::string, int> nameToIdx_;
    std::vector<int> anchors;
    std::set<std::string> whiteList_;
    int32_t verbose_;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ranks_;
    bool rankMatrixBuilt = false;

    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> compute_col_ranks(int col_idx) const {
        std::vector<uint32_t> indices(M_);
        std::iota(indices.begin(), indices.end(), 0);
        const auto& col = Q_.col(col_idx);
        std::sort(indices.begin(), indices.end(),
            [&](uint32_t a, uint32_t b) { return col(a) > col(b); });
        Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> rank_lookup(M_);
        for (uint32_t rank = 0; rank < M_; ++rank) {
            rank_lookup(indices[rank]) = rank;
        }
        return rank_lookup;
    }

    void buildRankMatrix() {
        ranks_.resize(M_, M_);
        tbb::parallel_for(
            tbb::blocked_range<int>(0, M_),
            [&](const tbb::blocked_range<int>& range) {
                for (int j = range.begin(); j < range.end(); ++j) {
                    ranks_.col(j) = compute_col_ranks(j);
                }
            }
        );
        rankMatrixBuilt = true;
        notice("Rank matrix built");
    }

    // Load co-occurrence counts (sparse) into Q_.
    void loadCooccurrenceMatrix(const std::string& filename, bool binary, uint32_t valueBytes = 8) {
        Q_.resize(M_, M_);
        Q_.setZero();
        if (binary) {
            std::ifstream ifs;
            ifs.open(filename, std::ios::binary);
            if (!ifs)
                error("Cannot open matrix: %s", filename.c_str());
            size_t npairs = 0;
            uint32_t f1, f2;
            double val;
            while (ifs.read(reinterpret_cast<char*>(&f1), sizeof(f1))) {
                ifs.read(reinterpret_cast<char*>(&f2), sizeof(f2));
                ifs.read(reinterpret_cast<char*>(&val), valueBytes);
                if (f1 < M_ && f2 < M_) {
                    Q_(f1, f2) = val;
                }
                npairs++;
                if (npairs % 1000000 == 0) {
                    notice("Read %zu pairs", npairs);
                }
            }
            ifs.close();
        } else {
            std::ifstream ifs;
            ifs.open(filename);
            if (!ifs)
                error("Cannot open matrix: %s", filename.c_str());
            std::string line;
            while (std::getline(ifs, line)) {
                std::istringstream iss(line);
                uint32_t f1, f2; double val;
                if (!(iss >> f1 >> f2 >> val)) continue;
                if (f1 < M_ && f2 < M_) {
                    Q_(f1, f2) = val;
                }
            }
            ifs.close();
        }
        normalizeRows();
        rankMatrixBuilt = false;
        notice("Loaded co-occurrence matrix");
    }

    void loadDenseCooccurrenceMatrix(const std::string& filename, bool binary, uint32_t valueBytes = 8) {
        notice("Loading dense matrix from %s", filename.c_str());
        if (binary) {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs)
                error("Cannot open dense matrix: %s", filename.c_str());
            uint32_t M_from_file;
            ifs.read(reinterpret_cast<char*>(&M_from_file), sizeof(M_from_file));
            if (!ifs || M_from_file == 0) {
                notice("Matrix dimension from file is 0 or unreadable. Nothing to load.");
                Q_.resize(0, 0);
                return;
            }
            M_ = M_from_file; // Set the class member M_
            Q_.resize(M_, M_);

            std::streamsize bytesToRead = static_cast<std::streamsize>(M_) * M_ * valueBytes;
            ifs.read(reinterpret_cast<char*>(Q_.data()), bytesToRead);
            if (ifs.gcount() != bytesToRead) {
                warning("Could not read the expected amount of data from dense binary matrix file: %s. Read %lld, expected %lld",
                    filename.c_str(), ifs.gcount(), bytesToRead);
            }
        } else { // TSV format
            std::ifstream ifs(filename);
            if (!ifs)
                error("Cannot open dense matrix: %s", filename.c_str());
            std::string line;
            std::vector<std::vector<double>> data;
            size_t M_from_file = 0;
            while (std::getline(ifs, line)) {
                std::istringstream iss(line);
                std::vector<double> row;
                double val;
                while (iss >> val) {
                    row.push_back(val);
                }
                if (!row.empty()) {
                    if (M_from_file == 0) {
                        M_from_file = row.size();
                    } else if (row.size() != M_from_file) {
                        warning("Inconsistent number of columns in dense TSV file: %s. Expected %zu, got %zu on line %zu.",
                            filename.c_str(), M_from_file, row.size(), data.size() + 1);
                    }
                    data.push_back(row);
                }
            }
            if (data.empty()) {
                notice("Dense TSV matrix file is empty: %s", filename.c_str());
                Q_.resize(0, 0);
                return;
            }
            if (data.size() != M_from_file) {
                warning("Number of rows (%zu) does not match number of columns (%zu) in dense TSV file: %s",
                    data.size(), M_from_file, filename.c_str());
            }
            M_ = M_from_file;
            Q_.resize(M_, M_);
            for (Eigen::Index i = 0; i < M_; ++i) {
                for (Eigen::Index j = 0; j < M_; ++j) {
                    if (i < data.size() && j < data[i].size()) {
                       Q_(i, j) = data[i][j];
                    } else {
                       Q_(i, j) = 0;
                    }
                }
            }
        }
        normalizeRows();
        rankMatrixBuilt = false;
        notice("Loaded dense co-occurrence matrix of size %u x %u", M_, M_);
    }

    void loadGeneInfo(const std::string& filename, int minUsedCount) {
        std::ifstream ifs(filename);
        if (!ifs)
            error("Cannot open gene info file: %s", filename.c_str());
        std::string line;
        M_ = 0;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            uint32_t idx, ct;
            std::string name;
            if (!(iss >> idx >> name >> ct)) {
                warning("Invalid line in gene info file: %s", line.c_str());
                continue;
            }
            if (idx > M_) {
                M_ = idx;
            }
            features_[idx] = {name, ct};
            if (ct > minUsedCount) {
                validIndex_.push_back(idx);
            } else if (whiteList_.count(name) > 0) {
                validIndex_.push_back(idx);
            }

            nameToIdx_[name] = idx;
        }
        M_++;
        if (M_ != features_.size()) {
            error("Feature information must cover all and only the features in the co-occurrence matrix (%d, %d)", M_, features_.size());
        }
        validGene_.resize(M_, false);
        for (auto i : validIndex_) {
            validGene_[i] = true;
        }
        notice("Load %d features", M_);
    }

    // Normalize each row of Q_ to sum to 1.
    void normalizeRows() {
        rowWeights_ = Q_.rowwise().sum();
        for (int i = 0; i < M_; ++i)
            if (rowWeights_(i) > 0) Q_.row(i) /= rowWeights_(i);
        rowWeights_ /= rowWeights_.sum();
    }

};
