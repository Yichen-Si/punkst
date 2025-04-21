#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdint>
#include <unordered_set>
#include "error.hpp"


class MarkerSelector {
public:
    // Construct with known matrix dimension M and weighting mode.
    MarkerSelector(const std::string& featureFile, const std::string& mtxFile, bool binary, int valueBytes, int minCount = 1, int32_t verbose = 0) : verbose_(verbose) {
        loadGeneInfo(featureFile, minCount);
        loadCooccurrenceMatrix(mtxFile, binary, valueBytes);
    }

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

    // Select K anchors, optionally fixing a subset
    void selectMarkers(
        int K, std::vector<std::string>& selectedAnchors) {

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
        // Build basis from the fixed anchors
        std::vector<Eigen::VectorXd> basis;
        for (int k = 0; k < F; ++k) {
            Eigen::VectorXd v = Q_.row(anchors[k]);
            for (auto &b : basis) {
                v -= b.dot(v) * b;
            }
            if (v.norm() > 1e-8) {
                basis.push_back(v.normalized());
            }
        }
        F = (int)basis.size();
        if (F > 0)
            notice("Received %d valid fixed anchors", F);
        if (F >= K)
            error("More than K fixed anchors provided");

        // Greedy select K anchors
        for (int k = F; k < K; ++k) {
            double bestD2 = -1;
            int    bestIdx = -1;
            for (auto i : validIndex_) {
                if (!validGene_[i] || fixedSet.count(i)) continue;
                Eigen::VectorXd r = Q_.row(i).transpose();
                // project r onto span(basis):
                Eigen::VectorXd proj = Eigen::VectorXd::Zero(r.size());
                for (auto &b : basis) {
                    proj.noalias() += b.dot(r) * b;
                }
                double d2 = (r - proj).squaredNorm();
                if (d2 > bestD2) {
                    bestD2 = d2;
                    bestIdx = i;
                }
            }
            anchors.push_back(bestIdx);
            fixedSet.insert(bestIdx);
            Eigen::VectorXd v = Q_.row(bestIdx).transpose();
            for (auto &b : basis) {
                v.noalias() -= b.dot(v) * b;
            }
            if (v.norm() > 1e-8) {
                basis.push_back(v.normalized());
            } else {
                warning("The matrix may have rank < %d", K);
            }
            if (verbose_ > 0) {
                notice("Selected %d-th anchor: %s (%d)", k, features_[bestIdx].name.c_str(), features_[bestIdx].totalOcc);
            }
        }

        // Refine
        for (int t = F; t < K; ++t) {
            fixedSet.clear();
            // Build basis excluding anchors[t]
            std::vector<Eigen::VectorXd> vecs;
            for (int j = 0; j < K; ++j) if (j != t) {
                Eigen::VectorXd v = Q_.row(anchors[j]).transpose();
                for (auto &b : vecs) {
                    v.noalias() -= b.dot(v) * b;
                }
                if (v.norm() > 1e-8) {
                    vecs.push_back(v.normalized());
                } else {
                    vecs.push_back(Eigen::VectorXd::Zero(M_));
                }
                fixedSet.insert(anchors[j]);
            }
            // Find farthest eligible gene
            double bestD2 = -1.0;
            int bestI = anchors[t];
            for (auto i : validIndex_) {
                if (!validGene_[i] || fixedSet.count(i)) continue;
                Eigen::VectorXd qi = Q_.row(i).transpose();
                Eigen::VectorXd proj = Eigen::VectorXd::Zero(M_);
                for (auto &b : vecs) {
                    proj.noalias() += b.dot(qi) * b;
                }
                double d2 = (qi - proj).squaredNorm();
                if (d2 > bestD2) {
                    bestD2 = d2;
                    bestI = i;
                }
            }
            if (verbose_ > 0 && bestI != anchors[t]) {
                notice("Replaced %d-th anchor %s with %s", t, features_[anchors[t]].name.c_str(), features_[bestI].name.c_str());
            }
            anchors[t] = bestI;
        }
        selectedAnchors.resize(K);
        for (int i = 0; i < K; ++i) {
            selectedAnchors[i] = features_[anchors[i]].name;
        }
    }

    std::vector<markerSetInfo> findNeighborsToAnchors(int32_t m, double maxRankFraction = -1) {
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
        uint32_t k = 0;
        for (int i : anchors) {
            // collect (feature, score)
            std::vector<std::pair<int,int>> cand;
            cand.reserve(M_);
            for (int j = 0; j < M_; ++j) {
                if (j == i) continue;
                int sc = std::max(rank[i][j], rank[j][i]);
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
                if (p.second > maxRank) break;
                res.neighbors.emplace_back(
                    features_[p.first].name,
                    features_[p.first].totalOcc,
                    Q_(i, p.first),
                    Q_(p.first, i),
                    rank[i][p.first],
                    rank[p.first][i]
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
            k++;
            result.push_back(std::move(res));
        }
        return result;
    }

private:
    struct featureInfo {
        std::string name;
        uint64_t totalOcc;
        uint64_t totalNeighbor;
    };

    uint32_t M_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q_;
    std::unordered_map<uint32_t, featureInfo>  features_;
    std::vector<uint32_t>     validIndex_;
    std::vector<bool>         validGene_;
    std::unordered_map<std::string, int> nameToIdx_;
    std::vector<int> anchors;
    std::vector<std::vector<uint32_t>> rank;
    int32_t verbose_;
    bool rankMatrixBuilt = false;

    void buildRankMatrix() {
        for (int i = 0; i < M_; ++i) {
            std::vector<uint32_t> ranki(M_);
            std::iota(ranki.begin(), ranki.end(), 0);
            std::sort(ranki.begin(), ranki.end(),
                [this, i](int a, int b) { return Q_(i, a) > Q_(i, b); });
            if (verbose_ > 1 || (verbose_ > 0 && i % 1000 == 0)) {
                notice("Ranked %d-th feature %s", i, features_[i].name.c_str());
            }
            rank.emplace_back(std::move(ranki));
        }
        rankMatrixBuilt = true;
        notice("Rank matrix built");
    }

    // Load co-occurrence counts (binary or TSV) into Q_.
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

    void loadGeneInfo(const std::string& filename, int minUsedCount) {
        std::ifstream ifs(filename);
        if (!ifs)
            error("Cannot open gene info file: %s", filename.c_str());
        std::string line;
        M_ = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            uint32_t idx, raw, tot, neigh;
            std::string name;
            if (!(iss >> idx >> name >> raw >> tot >> neigh)) {
                warning("Invalid line in gene info file: %s", line.c_str());
                continue;
            }
            if (idx > M_) {
                M_ = idx;
            }
            features_[idx] = {name, tot, neigh};
            if (tot > minUsedCount) {
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
        Eigen::VectorXd rowSums = Q_.rowwise().sum();
        for (int i = 0; i < M_; ++i)
            if (rowSums(i) > 0) Q_.row(i) /= rowSums(i);
    }

};
