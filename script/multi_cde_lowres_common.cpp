#include "multi_cde_lowres_common.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <unordered_set>

#include "lda.hpp"
#include "utils.h"

LowresPoisLink LowresScaleOptions::parsedLink() const {
    if (link == "log") return LowresPoisLink::Log;
    if (link == "log1p") return LowresPoisLink::Log1p;
    error("Unknown link: %s (expected log or log1p)", link.c_str());
    return LowresPoisLink::Log;
}

bool LowresScaleOptions::useLog1p() const {
    return parsedLink() == LowresPoisLink::Log1p;
}

double LowresScaleOptions::linkScale() const {
    return cScale > 0.0 ? cScale : sizeFactor;
}

void LowresScaleOptions::validate(bool usesLdaUncertainty) const {
    (void)parsedLink();
    if (cScale <= 0.0 && sizeFactor <= 0.0) {
        error("Size factor must be positive when --c is not set");
    }
    if (cScale > 0.0 && usesLdaUncertainty) {
        error("--propagate-uncertainty requires size-factor scaling and cannot be combined with --c");
    }
}

namespace {

void readLowresContrastFile(const std::string& contrastFile,
        std::vector<std::string>& inFile,
        std::vector<std::string>& metaFile,
        std::vector<std::vector<int32_t>>& contrasts,
        std::vector<std::string>& contrastNames) {
    std::ifstream ifs(contrastFile);
    if (!ifs.is_open()) {
        error("Cannot open contrast file: %s", contrastFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    if (!std::getline(ifs, line)) {
        error("Contrast file is empty: %s", contrastFile.c_str());
    }
    split(tokens, "\t ", line, UINT_MAX, true, true, true);
    if (tokens.size() < 3) {
        error("Contrast file must have >= 3 columns (in-data, in-meta, contrast...): %s",
            contrastFile.c_str());
    }
    const int32_t C = static_cast<int32_t>(tokens.size() - 2);
    contrasts.assign(C, std::vector<int32_t>{});
    contrastNames.resize(C);
    std::unordered_set<std::string> seen_names;
    for (int32_t c = 0; c < C; ++c) {
        const std::string name = tokens[c + 2];
        if (name.empty()) {
            warning("Contrast %d has an empty header name; using index instead", c);
            contrastNames[c] = std::to_string(c);
        } else {
            if (!seen_names.insert(name).second) {
                error("Contrast name '%s' appears more than once in header", name.c_str());
            }
            contrastNames[c] = name;
        }
    }

    int32_t n_samples = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        split(tokens, "\t ", line, UINT_MAX, true, true, true);
        if (tokens.size() != static_cast<size_t>(C + 2)) {
            error("Contrast file row has %zu columns, expected %d: %s",
                tokens.size(), C + 2, line.c_str());
        }
        inFile.push_back(tokens[0]);
        metaFile.push_back(tokens[1]);
        for (int32_t c = 0; c < C; ++c) {
            int32_t y = 0;
            if (!str2int32(tokens[c + 2], y)) {
                error("Contrast value must be -1, 0, or 1: %s", tokens[c + 2].c_str());
            }
            if (y < -1 || y > 1) {
                error("Contrast value must be -1, 0, or 1: %d", y);
            }
            contrasts[c].push_back(y);
        }
        n_samples++;
    }
    if (n_samples == 0) {
        error("Contrast file has a header but no samples: %s", contrastFile.c_str());
    }
    for (int32_t c = 0; c < C; ++c) {
        if (contrasts[c].size() != static_cast<size_t>(n_samples)) {
            error("Contrast %s has %zu labels, expected %d samples",
                contrastNames[c].c_str(), contrasts[c].size(), n_samples);
        }
        int32_t n_neg = 0;
        int32_t n_pos = 0;
        for (int32_t s = 0; s < n_samples; ++s) {
            const int32_t y = contrasts[c][s];
            if (y < 0) {
                n_neg++;
            } else if (y > 0) {
                n_pos++;
            }
        }
        if (n_neg == 0 || n_pos == 0) {
            error("Contrast %s must have at least one sample in each group", contrastNames[c].c_str());
        }
    }
}

} // namespace

LowresContrastDesign loadLowresContrastDesign(
    const std::string& contrastFile,
    const std::vector<std::string>& inFile,
    const std::vector<std::string>& metaFile,
    const std::vector<std::string>& dataLabels) {
    LowresContrastDesign design;
    design.dataLabels = dataLabels;
    if (!contrastFile.empty()) {
        if (!inFile.empty() || !metaFile.empty()) {
            error("--contrast cannot be combined with --in-data/--in-meta");
        }
        readLowresContrastFile(contrastFile, design.inFile, design.metaFile,
            design.contrasts, design.contrastNames);
        notice("Read %d contrasts from %s", design.nContrasts(), contrastFile.c_str());
    } else {
        design.inFile = inFile;
        design.metaFile = metaFile;
    }
    if (design.inFile.empty() || design.metaFile.empty() ||
            design.inFile.size() != design.metaFile.size()) {
        error("Either --contrast or both --in-data and --in-meta must be specified");
    }

    const int32_t G = design.nSamples();
    if (contrastFile.empty()) {
        design.dataLabels.resize(G);
        for (int32_t g = 0; g < G; ++g) {
            if (design.dataLabels[g].empty()) design.dataLabels[g] = std::to_string(g);
        }
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t l = g + 1; l < G; ++l) {
                design.contrasts.push_back(std::vector<int32_t>(G, 0));
                design.contrasts.back()[g] = -1;
                design.contrasts.back()[l] = 1;
                design.contrastNames.push_back(design.dataLabels[g] + "v" + design.dataLabels[l]);
            }
        }
        notice("Will perform all pairwise contrasts (%d) among %d samples",
            design.nContrasts(), G);
    }
    if (design.contrasts.empty()) {
        error("No contrasts defined");
    }
    return design;
}

LowresInputData loadLowresInputData(
    const std::string& modelFile,
    int32_t seed,
    int32_t nThreads,
    const LowresContrastDesign& design,
    const LowresScaleOptions& scale,
    int32_t minCount,
    int32_t debug,
    RowMajorMatrixXd& eta0) {
    LowresInputData out;
    LatentDirichletAllocation lda(modelFile, seed, nThreads);
    Eigen::MatrixXd beta = lda.get_model();
    rowNormalizeInPlace(beta);

    const double linkScale = scale.linkScale();
    if (scale.useLog1p()) {
        eta0 = (beta.array() * linkScale + 1.0).log();
    } else {
        eta0 = (beta.array().max(1e-8) * linkScale).log();
    }
    out.K = lda.get_n_topics();
    out.M = lda.get_n_features();
    out.factorNames = lda.topic_names_;
    out.featureNames = lda.feature_names_;

    const int32_t G = design.nSamples();
    std::vector<HexReader> readers;
    readers.reserve(G);
    for (int32_t g = 0; g < G; ++g) {
        readers.emplace_back(design.metaFile[g]);
        readers.back().setFeatureIndexRemap(out.featureNames);
    }

    out.nUnits.assign(G, 0);
    out.docsT.resize(G);
    out.offsets.assign(G + 1, 0);
    out.cThetaSums.assign(G, VectorXd::Zero(out.K));
    std::vector<std::vector<double>> cvecs(G);
    std::vector<RowMajorMatrixXd> thetas(G);
    const int32_t batchSize = 1024;
    for (int32_t g = 0; g < G; ++g) {
        SparseObsMinibatchReader batch_reader(design.inFile[g], readers[g],
            minCount, scale.sizeFactor, scale.cScale, debug);
        std::vector<SparseObs> docs;
        RowMajorMatrixXd theta_g(0, out.K);
        int32_t theta_capacity = 0;
        int32_t nUnits_local = 0;
        out.docsT[g].assign(out.M, Document{});
        while (true) {
            int32_t n_batch = batch_reader.readBatch(docs, nullptr, batchSize);
            if (n_batch == 0) {break;}
            RowMajorMatrixXd theta_batch = lda.transform(docs);
            rowNormalizeInPlace(theta_batch);
            if (nUnits_local + n_batch > theta_capacity) {
                theta_capacity = std::max(theta_capacity + 2 * batchSize, nUnits_local + n_batch);
                theta_g.conservativeResize(theta_capacity, out.K);
            }
            theta_g.block(nUnits_local, 0, n_batch, out.K) = theta_batch;
            cvecs[g].reserve(static_cast<size_t>(nUnits_local + n_batch));
            for (int32_t i = 0; i < n_batch; ++i) {
                cvecs[g].push_back(docs[i].c);
                const Document& doc = docs[i].doc;
                for (size_t t = 0; t < doc.ids.size(); ++t) {
                    const uint32_t m = doc.ids[t];
                    out.docsT[g][m].ids.push_back(nUnits_local + i);
                    out.docsT[g][m].cnts.push_back(doc.cnts[t]);
                }
            }
            nUnits_local += n_batch;
        }
        out.nUnits[g] = nUnits_local;
        notice("Read %d units from the %d-th data file", out.nUnits[g], g);
        theta_g.conservativeResize(out.nUnits[g], out.K);
        if (out.nUnits[g] == 0) {
            warning("No units passed the --min-count filter for the %d-th data file", g);
        } else {
            Eigen::Map<const Eigen::ArrayXd> cvec_g(cvecs[g].data(), out.nUnits[g]);
            out.cThetaSums[g] = (theta_g.array().colwise() * cvec_g).matrix().colwise().sum();
        }
        thetas[g] = std::move(theta_g);
        out.offsets[g + 1] = out.offsets[g] + out.nUnits[g];
    }

    out.N_total = out.offsets.back();
    out.A_all.resize(out.N_total, out.K);
    out.cvec_all.resize(out.N_total);
    for (int32_t g = 0; g < G; ++g) {
        const int32_t ng = out.nUnits[g];
        const int32_t offset = out.offsets[g];
        out.A_all.block(offset, 0, ng, out.K) = thetas[g];
        out.cvec_all.segment(offset, ng) =
            Eigen::Map<const VectorXd>(cvecs[g].data(), ng);
    }
    notice("Total %d units across all %d datasets", out.N_total, G);
    return out;
}

LowresContrastData buildLowresContrastData(
    const LowresInputData& data,
    const std::vector<int32_t>& contrast) {
    LowresContrastData out;
    out.xvec = VectorXd::Zero(data.N_total);
    out.cvecMasked = VectorXd::Zero(data.N_total);
    out.aSum = VectorXd::Zero(data.K);
    for (int32_t g = 0; g < static_cast<int32_t>(contrast.size()); ++g) {
        if (contrast[g] == 0) {
            continue;
        }
        out.n += data.nUnits[g];
        const int32_t ng = data.nUnits[g];
        const int32_t offset = data.offsets[g];
        out.xvec.segment(offset, ng) = VectorXd::Constant(ng, contrast[g]);
        out.cvecMasked.segment(offset, ng) = data.cvec_all.segment(offset, ng);
        out.aSum += data.cThetaSums[g];
    }
    out.cSum = out.cvecMasked.sum();
    return out;
}

bool buildLowresFeatureObs(
    const LowresInputData& data,
    const std::vector<int32_t>& contrast,
    int32_t feature,
    int32_t minUnits,
    int32_t minCountFeature,
    LowresFeatureObs& out) {
    out = LowresFeatureObs{};
    out.feature = feature;
    for (int32_t g = 0; g < static_cast<int32_t>(contrast.size()); ++g) {
        if (contrast[g] == 0) {continue;}
        const auto& doc = data.docsT[g][feature];
        const double ysum = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
        out.nnz += static_cast<int32_t>(doc.ids.size());
        if (contrast[g] < 0) {
            out.ysum0 += ysum;
        } else {
            out.ysum1 += ysum;
        }
    }
    if (out.nnz < minUnits) {return false;}
    out.totalCount = static_cast<int32_t>(out.ysum0 + out.ysum1);
    if (out.totalCount < minCountFeature) {return false;}

    out.y.ids.reserve(out.nnz);
    out.y.cnts.reserve(out.nnz);
    for (int32_t g = 0; g < static_cast<int32_t>(contrast.size()); ++g) {
        if (contrast[g] == 0) {continue;}
        const auto& doc = data.docsT[g][feature];
        const int32_t offset = data.offsets[g];
        for (size_t t = 0; t < doc.ids.size(); ++t) {
            out.y.ids.push_back(static_cast<uint32_t>(offset) + doc.ids[t]);
            out.y.cnts.push_back(doc.cnts[t]);
        }
    }
    return true;
}

std::vector<int32_t> shuffledFeatureOrder(int32_t M, int32_t seed) {
    std::vector<int32_t> perm_idx(M);
    for (int32_t j = 0; j < M; ++j) {perm_idx[j] = j;}
    std::mt19937 rng(seed);
    std::shuffle(perm_idx.begin(), perm_idx.end(), rng);
    return perm_idx;
}

void writeLowresEta0(
    const std::string& path,
    const RowMajorMatrixXd& eta0,
    bool useLog1p,
    const std::vector<std::string>& featureNames,
    const std::vector<std::string>& factorNames) {
    RowMajorMatrixXd eta_out;
    if (useLog1p) {
        eta_out = eta0.transpose().array().exp() - 1.0;
    } else {
        eta_out = eta0.transpose().array().exp();
    }
    write_matrix_to_file(path, eta_out, 4, true, featureNames, "Feature", &factorNames);
}
