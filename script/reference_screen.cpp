#include "punkst.h"
#include "utils.h"

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

using Matrix = Eigen::MatrixXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

struct TargetMoments {
    std::vector<std::string> topics;
    Vector a;
    Matrix G;
    std::vector<double> T;
    int64_t units = 0;
    int64_t skippedUnits = 0;
    int64_t activeLoadings = 0;
    double discardedMass = 0.0;
    double maxDiscardedMass = 0.0;
};

size_t packedTripleIndex(int i, int j, int k) {
    return static_cast<size_t>(k) * (k + 1) * (k + 2) / 6
        + static_cast<size_t>(j) * (j + 1) / 2 + i;
}

size_t symmetricTripleIndex(int i, int j, int k) {
    if (i > j) std::swap(i, j);
    if (j > k) std::swap(j, k);
    if (i > j) std::swap(i, j);
    return packedTripleIndex(i, j, k);
}

struct ProjectedTarget {
    Matrix B;
    std::vector<Vector> means;
    std::vector<Matrix> seconds;
    double meanScale = 1.0;
    double secondScale = 1.0;
};

struct Evaluation {
    double meanLoss = 0.0;
    double secondLoss = 0.0;
    Vector gradient;
    double value() const { return meanLoss + secondLoss; }
};

struct FactorSolve {
    double lower = 0.0;
    double upper = 0.0;
    double meanLoss = 0.0;
    double secondLoss = 0.0;
    double trace = 0.0;
    double gap = 0.0;
    int iterations = 0;
    bool converged = false;
    Vector weights;
};

struct ScoreRow {
    std::string reference;
    std::string path;
    std::string status;
    double tau = 0.0;
    std::array<double, 4> lower = {NAN, NAN, NAN, NAN};
    std::array<double, 4> upper = {NAN, NAN, NAN, NAN};
    std::array<int, 4> rank = {0, 0, 0, 0};
    double maxGap = std::numeric_limits<double>::quiet_NaN();
    int worstFactor = -1;
    int sharedFeatures = 0;
    int referenceFeatures = 0;
    int labels = 0;
    int iterations = 0;
    int selectedFactors = 0;
};

struct ReferenceResult {
    std::vector<ScoreRow> scores;
    std::string factorRows;
    std::string error;
    int sharedFeatures = 0;
    int labels = 0;
};

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * (values.size() - 1);
    const size_t left = static_cast<size_t>(position);
    const size_t right = std::min(left + 1, values.size() - 1);
    const double weight = position - left;
    return (1.0 - weight) * values[left] + weight * values[right];
}

std::vector<std::string> tabFields(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        const size_t end = line.find('\t', start);
        fields.push_back(line.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return fields;
}

TargetMoments readTargetMoments(const std::string& path,
        const std::vector<std::string>& topics, bool excludedFactors,
        double loadingPruneThreshold) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open target results: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty target results: " + path);
    const auto header = tabFields(line);
    const int K = static_cast<int>(topics.size());
    std::unordered_map<std::string, int> headerIndex;
    for (int j = 0; j < static_cast<int>(header.size()); ++j) {
        if (!headerIndex.emplace(header[j], j).second) {
            throw std::runtime_error("Duplicate target results header: " + header[j]);
        }
    }
    std::vector<int> columns(K);
    for (int k = 0; k < K; ++k) {
        const auto found = headerIndex.find(topics[k]);
        if (found == headerIndex.end()) {
            throw std::runtime_error("Target results lack dense topic column: " + topics[k]);
        }
        columns[k] = found->second;
    }
    TargetMoments out;
    out.topics = topics;
    out.a = Vector::Zero(K);
    out.G = Matrix::Zero(K, K);
    out.T.assign(static_cast<size_t>(K) * (K + 1) * (K + 2) / 6, 0.0);
    std::vector<std::string> tokens;
    int minColumns = *std::max_element(columns.begin(), columns.end());
    Vector theta(K);
    std::vector<int> active;
    active.reserve(K);
    int64_t rows = 0;
    notice("Reading target loadings (K=%d) from %s", K, path.c_str());
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        ++rows;
        if (rows % 50000 == 0)
            notice("Processed %d target units, skipped %d", rows, out.skippedUnits);
        split(tokens, "\t", line, UINT_MAX, true, false, true);
        if (static_cast<int>(tokens.size()) <= minColumns) {
            throw std::runtime_error("Wrong number of columns in target results row " + std::to_string(rows + 1));
        }
        double total = 0.0;
        double value;
        for (int k = 0; k < K; ++k) {
            if (!str2double(tokens[columns[k]], value) || !std::isfinite(value)
                    || value < 0.0) {
                throw std::runtime_error("Invalid topic weight in target results row "
                    + std::to_string(rows + 1));
            }
            theta[k] = value;
            total += value;
        }
        if (!(total > 0.0) || (excludedFactors && total < 0.5)) {
            ++out.skippedUnits;
            continue;
        }
        theta /= total;
        active.clear();
        double retainedMass = 0.0, discardedMass = 0.0;
        for (int k = 0; k < K; ++k) {
            if (theta[k] > loadingPruneThreshold) {
                active.push_back(k);
                retainedMass += theta[k];
            } else {
                discardedMass += theta[k];
            }
        }
        if (!(retainedMass > 0.0)) {
            throw std::runtime_error("No target loadings survive pruning in results row "
                + std::to_string(rows + 1));
        }
        out.discardedMass += discardedMass;
        out.maxDiscardedMass = std::max(out.maxDiscardedMass, discardedMass);
        out.activeLoadings += active.size();
        for (size_t j = 0; j < active.size(); ++j) {
            const int a = active[j];
            out.a[a] += theta[a];
            for (size_t l = j; l < active.size(); ++l) {
                const int b = active[l];
                const double pair = theta[a] * theta[b];
                out.G(a, b) += pair;
                if (a != b) out.G(b, a) += pair;
                for (size_t k = l; k < active.size(); ++k) {
                    const int c = active[k];
                    out.T[packedTripleIndex(a, b, c)] += pair * theta[c];
                }
            }
        }
        ++out.units;
    }
    if (out.units == 0) {
        throw std::runtime_error("No target units pass the retained factor loading filter");
    }
    out.a /= static_cast<double>(out.units);
    out.G /= static_cast<double>(out.units);
    for (double& value : out.T) value /= static_cast<double>(out.units);
    return out;
}

void checkMatrix(const RowMatrix& matrix, const std::string& path) {
    if (matrix.rows() == 0 || matrix.cols() == 0 || !matrix.allFinite()
            || (matrix.array() < 0.0).any()) {
        throw std::runtime_error("Matrix must be nonempty, finite, and nonnegative: " + path);
    }
}

void readReferenceMatrix(const std::string& path, RowMatrix& matrix,
        std::vector<std::string>& features, std::vector<std::string>& labels) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open reference matrix: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty reference matrix");
    const auto header = tabFields(line);
    if (header.size() < 3) throw std::runtime_error("Fewer than two reference labels");
    labels.assign(header.begin() + 1, header.end());
    for (const auto& label : labels) {
        if (trim(label).empty()) throw std::runtime_error("Empty reference label");
    }
    std::vector<double> values;
    values.reserve(1024 * labels.size());
    size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        const auto fields = tabFields(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("Wrong number of columns in reference row "
                + std::to_string(lineNumber));
        }
        if (trim(fields[0]).empty()) {
            throw std::runtime_error("Empty feature name in reference row "
                + std::to_string(lineNumber));
        }
        features.push_back(fields[0]);
        for (size_t j = 1; j < fields.size(); ++j) {
            double value;
            try {
                value = parse_scalar<double>(fields[j]);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid value in reference row "
                    + std::to_string(lineNumber) + ", column " + std::to_string(j + 1));
            }
            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error("Nonfinite or negative value in reference row "
                    + std::to_string(lineNumber) + ", column " + std::to_string(j + 1));
            }
            values.push_back(value);
        }
    }
    if (input.bad()) throw std::runtime_error("Cannot finish reading reference matrix: " + path);
    if (features.empty()) throw std::runtime_error("Reference matrix has no features");
    matrix.resize(features.size(), labels.size());
    std::copy(values.begin(), values.end(), matrix.data());
}

ProjectedTarget prepareReference(const RowMatrix& model,
        const std::vector<std::string>& modelFeatures,
        const TargetMoments& moments, const RowMatrix& reference,
        const std::vector<std::string>& refFeatures, int minSharedFeatures,
        int& sharedCount) {
    const int K = static_cast<int>(model.cols());
    const int C = static_cast<int>(reference.cols());
    std::unordered_map<std::string, int> modelIndex;
    modelIndex.reserve(modelFeatures.size());
    for (int m = 0; m < static_cast<int>(modelFeatures.size()); ++m) {
        if (!modelIndex.emplace(modelFeatures[m], m).second) {
            throw std::runtime_error("Duplicate target model feature: " + modelFeatures[m]);
        }
    }
    std::vector<bool> present(model.rows(), false);
    for (const auto& gene : refFeatures) {
        const auto found = modelIndex.find(gene);
        if (found != modelIndex.end()) present[found->second] = true;
    }
    std::vector<int> selected;
    selected.reserve(model.rows());
    std::vector<int> panelIndex(model.rows(), -1);
    for (int m = 0; m < model.rows(); ++m) {
        if (present[m]) {
            panelIndex[m] = static_cast<int>(selected.size());
            selected.push_back(m);
        }
    }
    sharedCount = static_cast<int>(selected.size());
    if (sharedCount < minSharedFeatures) {
        throw std::runtime_error("Only " + std::to_string(sharedCount)
            + " shared features; require at least " + std::to_string(minSharedFeatures));
    }
    Matrix V(sharedCount, K);
    Matrix X = Matrix::Zero(sharedCount, C);
    for (int m = 0; m < sharedCount; ++m) V.row(m) = model.row(selected[m]);
    for (int m = 0; m < reference.rows(); ++m) {
        const auto found = modelIndex.find(refFeatures[m]);
        if (found != modelIndex.end()) {
            X.row(panelIndex[found->second]) += reference.row(m);
        }
    }
    for (int k = 0; k < K; ++k) {
        const double sum = V.col(k).sum();
        if (!(sum > 0.0)) throw std::runtime_error("Target topic has zero mass on shared features");
        V.col(k) /= sum;
    }
    for (int c = 0; c < C; ++c) {
        const double sum = X.col(c).sum();
        if (!(sum > 0.0)) throw std::runtime_error("Reference label has zero mass on shared features");
        X.col(c) /= sum;
    }
    Matrix gram = V.transpose() * V;
    Eigen::SelfAdjointEigenSolver<Matrix> eig(gram);
    if (eig.info() != Eigen::Success) throw std::runtime_error("Target projection eigensolve failed");
    const double sigmaMax = std::sqrt(std::max(0.0, eig.eigenvalues().maxCoeff()));
    if (!(sigmaMax > 0.0)) throw std::runtime_error("Target projection has zero rank");
    const Matrix cross = V.transpose() * X;
    const int rank = static_cast<int>((eig.eigenvalues().array()
        > 1e-16 * sigmaMax * sigmaMax).count());
    Matrix F(rank, K), B(rank, C);
    int row = 0;
    for (int j = K - 1; j >= 0; --j) {
        const double sigma = std::sqrt(std::max(0.0, eig.eigenvalues()[j]));
        if (sigma <= 1e-8 * sigmaMax) continue;
        const double divisor = std::max(sigma, 0.1 * sigmaMax);
        F.row(row) = (sigma / divisor) * eig.eigenvectors().col(j).transpose();
        B.row(row) = eig.eigenvectors().col(j).transpose() * cross
            / (sigma * divisor);
        ++row;
    }
    ProjectedTarget out;
    out.B = std::move(B);
    out.means.resize(K);
    out.seconds.resize(K);
    double meanNorm = 0.0, secondNorm = 0.0;
    int active = 0;
    for (int k = 0; k < K; ++k) {
        if (!(moments.a[k] > 0.0)) continue;
        out.means[k] = F * (moments.G.col(k) / moments.a[k]);
        Matrix Tk(K, K);
        for (int j = 0; j < K; ++j) {
            for (int l = 0; l < K; ++l) {
                Tk(j, l) = moments.T[symmetricTripleIndex(k, j, l)]
                    / moments.a[k];
            }
        }
        out.seconds[k] = F * Tk * F.transpose();
        meanNorm += out.means[k].squaredNorm();
        secondNorm += out.seconds[k].squaredNorm();
        ++active;
    }
    out.meanScale = 1.0 / std::max(1e-12, meanNorm / active);
    out.secondScale = 1.0 / std::max(1e-12, secondNorm / active);
    return out;
}

Vector simplexProjection(const Vector& values, double mass) {
    Vector result = Vector::Zero(values.size());
    if (values.size() == 0 || mass <= 0.0) return result;
    std::vector<double> ordered(values.data(), values.data() + values.size());
    std::sort(ordered.begin(), ordered.end(), std::greater<double>());
    double cumulative = 0.0, threshold = 0.0;
    for (int j = 0; j < values.size(); ++j) {
        cumulative += ordered[j];
        const double candidate = (cumulative - mass) / (j + 1);
        if (ordered[j] > candidate) threshold = candidate;
    }
    result = (values.array() - threshold).max(0.0);
    return result;
}

Vector projectWeights(const Vector& values, int C, double tau) {
    Vector result = simplexProjection(values, 1.0);
    if (result.head(C).sum() >= tau - 1e-12) return result;
    result.head(C) = simplexProjection(values.head(C), tau);
    if (values.size() > C) {
        result.tail(values.size() - C) =
            simplexProjection(values.tail(values.size() - C), 1.0 - tau);
    }
    return result;
}

class FactorObjective {
public:
    FactorObjective(const Matrix& profiles, const Vector& mean,
            const Matrix& second, double meanScale, double secondScale)
        : B(profiles), mu(mean), S(second), gamma1(meanScale), gamma2(secondScale),
          C(static_cast<int>(profiles.cols())),
          variables(C * (C + 1) / 2) {}

    Evaluation evaluate(const Vector& weights, bool gradient) const {
        Matrix Q = Matrix::Zero(C, C);
        Q.diagonal() = weights.head(C);
        int index = C;
        for (int i = 0; i < C; ++i) {
            for (int j = i + 1; j < C; ++j) {
                Q(i, j) = Q(j, i) = 0.5 * weights[index++];
            }
        }
        const Vector rMean = B * Q.rowwise().sum().eval() - mu;
        const Matrix rSecond = B * Q * B.transpose() - S;
        Evaluation out;
        out.meanLoss = gamma1 * rMean.squaredNorm();
        out.secondLoss = gamma2 * rSecond.squaredNorm();
        if (gradient) {
            const Vector first = B.transpose() * rMean;
            const Matrix second = B.transpose() * rSecond * B;
            out.gradient.resize(variables);
            for (int i = 0; i < C; ++i) {
                out.gradient[i] = 2.0 * gamma1 * first[i]
                    + 2.0 * gamma2 * second(i, i);
            }
            index = C;
            for (int i = 0; i < C; ++i) {
                for (int j = i + 1; j < C; ++j) {
                    out.gradient[index++] = gamma1 * (first[i] + first[j])
                        + 2.0 * gamma2 * second(i, j);
                }
            }
        }
        return out;
    }

    int count() const { return variables; }
private:
    const Matrix& B;
    const Vector& mu;
    const Matrix& S;
    double gamma1, gamma2;
    int C, variables;
};

double linearMinimum(const Vector& gradient, int C, double tau) {
    const double diagMin = gradient.head(C).minCoeff();
    if (gradient.size() == C) return diagMin;
    const double offMin = gradient.tail(gradient.size() - C).minCoeff();
    return diagMin <= offMin ? diagMin : tau * diagMin + (1.0 - tau) * offMin;
}

FactorSolve solveFactor(const FactorObjective& objective, int C, double tau,
        const Vector& initial, int maxIter, double tolerance) {
    Vector x = projectWeights(initial, C, tau);
    Vector y = x;
    double f = objective.evaluate(x, false).value();
    double momentum = 1.0, L = 1.0;
    FactorSolve result;
    for (int iter = 0; iter < maxIter; ++iter) {
        Evaluation atY = objective.evaluate(y, true);
        Vector next;
        double nextValue = 0.0;
        bool accepted = false;
        for (int trial = 0; trial < 60; ++trial) {
            next = projectWeights(y - atY.gradient / L, C, tau);
            nextValue = objective.evaluate(next, false).value();
            const Vector delta = next - y;
            if (nextValue <= atY.value() + atY.gradient.dot(delta)
                    + 0.5 * L * delta.squaredNorm() + 1e-12) {
                accepted = true;
                break;
            }
            L *= 2.0;
        }
        if (!accepted || !std::isfinite(nextValue)) {
            throw std::runtime_error("Projected-gradient line search failed");
        }
        if (nextValue > f + 1e-12) {
            y = x;
            momentum = 1.0;
            atY = objective.evaluate(y, true);
            accepted = false;
            for (int trial = 0; trial < 60; ++trial) {
                next = projectWeights(y - atY.gradient / L, C, tau);
                nextValue = objective.evaluate(next, false).value();
                const Vector delta = next - y;
                if (nextValue <= atY.value() + atY.gradient.dot(delta)
                        + 0.5 * L * delta.squaredNorm() + 1e-12) {
                    accepted = true;
                    break;
                }
                L *= 2.0;
            }
            if (!accepted || !std::isfinite(nextValue)) {
                throw std::runtime_error("Projected-gradient restart failed");
            }
        }
        Vector previous = x;
        x = std::move(next);
        f = nextValue;
        const double newMomentum = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * momentum * momentum));
        y = x + ((momentum - 1.0) / newMomentum) * (x - previous);
        momentum = newMomentum;
        L = std::max(1e-12, L * 0.8);
        result.iterations = iter + 1;
        if (iter % 10 == 9 || iter + 1 == maxIter) {
            const Evaluation current = objective.evaluate(x, true);
            const double gap = std::max(0.0, current.gradient.dot(x)
                - linearMinimum(current.gradient, C, tau));
            if (gap <= tolerance * std::max(1.0, current.value())) {
                result.converged = true;
                break;
            }
        }
    }
    const Evaluation end = objective.evaluate(x, true);
    result.weights = std::move(x);
    result.meanLoss = end.meanLoss;
    result.secondLoss = end.secondLoss;
    result.upper = end.value();
    result.gap = std::max(0.0, end.gradient.dot(result.weights)
        - linearMinimum(end.gradient, C, tau));
    result.lower = std::max(0.0, result.upper - result.gap);
    result.trace = result.weights.head(C).sum();
    result.converged = result.gap <= tolerance * std::max(1.0, result.upper);
    return result;
}

} // namespace

int32_t cmdReferenceScreen(int argc, char** argv) {
    std::string modelPath, resultsPath, outPrefix, referenceList;
    std::vector<std::string> references, referenceIds;
    std::vector<double> tauValues;
    int threads = 1, maxIter = 300, verbose = 0, minSharedFeatures = 50;
    double tolerance = 1e-5, factorMassThreshold = 0.999;
    double loadingPruneThreshold = 1e-6;
    ParamList pl;
    pl.add_option("model", "Fitted LDA .model.tsv", modelPath, true)
      .add_option("results", "Dense LDA .results.tsv", resultsPath, true)
      .add_option("references", "Reference feature-by-profile TSV files", references)
      .add_option("reference-ids", "IDs matching --references in order", referenceIds)
      .add_option("reference-list", "TSV with reference ID and path in its first two columns", referenceList)
      .add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("tau-values", "Concentration thresholds in [0,1]", tauValues)
      .add_option("factor-mass-threshold", "Fraction of target model mass to retain (0,1]", factorMassThreshold)
      .add_option("min-shared-features", "Minimum distinct features shared with the target for a panel to be scored (default 50; at least 2)", minSharedFeatures)
      .add_option("loading-prune-threshold", "Drop normalized target loadings at or below this value [0,1); 0 keeps all positive loadings", loadingPruneThreshold)
      .add_option("threads", "Total threads available for reference screening", threads)
      .add_option("verbose", "Report progress every this many screened panels (0 disables progress notices)", verbose)
      .add_option("max-iter", "Maximum projected-gradient iterations per solve", maxIter)
      .add_option("tol", "Relative convex optimality gap tolerance", tolerance);
    try {
        pl.readArgs(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << '\n';
        pl.print_help_noexit();
        return 1;
    }
    if (!referenceIds.empty() && referenceIds.size() != references.size()) {
        throw std::runtime_error("--reference-ids must have the same number of values as --references");
    }
    std::vector<std::string> ids;
    ids.reserve(references.size());
    for (size_t i = 0; i < references.size(); ++i) {
        ids.push_back(referenceIds.empty() ? std::to_string(i) : referenceIds[i]);
    }
    if (!referenceList.empty()) {
        std::ifstream list(referenceList);
        if (!list) throw std::runtime_error("Cannot open reference list: " + referenceList);
        std::string line;
        size_t lineNumber = 0;
        while (std::getline(list, line)) {
            ++lineNumber;
            const std::string stripped = trim(line);
            if (stripped.empty() || stripped[0] == '#') continue;
            const auto cells = tabFields(line);
            if (cells.size() < 2) {
                throw std::runtime_error("Reference list row " + std::to_string(lineNumber)
                    + " must have at least two tab-separated columns");
            }
            const std::string id = trim(cells[0]);
            const std::string path = trim(cells[1]);
            if (id.empty() || path.empty()) {
                throw std::runtime_error("Reference list row " + std::to_string(lineNumber)
                    + " has an empty ID or path");
            }
            ids.push_back(id);
            references.push_back(path);
        }
    }
    if (references.empty()) throw std::runtime_error("Supply --references or --reference-list");
    if (tauValues.empty()) tauValues = {0.5, 0.75, 1.0};
    for (double tau : tauValues) {
        if (!std::isfinite(tau) || tau < 0.0 || tau > 1.0) {
            throw std::runtime_error("--tau-values must lie in [0,1]");
        }
    }
    if (threads < 1 || maxIter < 1 || !(tolerance > 0.0)
            || !std::isfinite(tolerance)) {
        throw std::runtime_error("Require positive threads and max-iter, and finite positive tol");
    }
    if (minSharedFeatures < 2) {
        throw std::runtime_error("--min-shared-features must be at least 2");
    }
    if (!std::isfinite(factorMassThreshold) || !(factorMassThreshold > 0.0)
            || factorMassThreshold > 1.0) {
        throw std::runtime_error("--factor-mass-threshold must lie in (0,1]");
    }
    if (!std::isfinite(loadingPruneThreshold) || loadingPruneThreshold < 0.0
            || loadingPruneThreshold >= 1.0) {
        throw std::runtime_error("--loading-prune-threshold must lie in [0,1)");
    }
    std::sort(tauValues.begin(), tauValues.end());
    tauValues.erase(std::unique(tauValues.begin(), tauValues.end()), tauValues.end());
    std::unordered_set<std::string> seenPaths;
    std::unordered_set<std::string> seenIds;
    for (size_t i = 0; i < references.size(); ++i) {
        if (ids[i].empty()) throw std::runtime_error("Reference IDs must be nonempty");
        if (!seenIds.insert(ids[i]).second) {
            throw std::runtime_error("Duplicate reference ID: " + ids[i]);
        }
        if (!seenPaths.insert(references[i]).second) {
            throw std::runtime_error("Duplicate reference path: " + references[i]);
        }
    }

    RowMatrix model;
    std::vector<std::string> modelFeatures, topics;
    read_matrix_from_file<double>(modelPath, model, &modelFeatures, &topics);
    checkMatrix(model, modelPath);
    if (topics.size() != static_cast<size_t>(model.cols())) {
        throw std::runtime_error("Invalid target model header");
    }
    const Vector modelMass = model.colwise().sum().transpose();
    const double totalModelMass = modelMass.sum();
    if (!(totalModelMass > 0.0)) {
        throw std::runtime_error("Target model has zero total mass");
    }
    std::vector<int> modelOrder(model.cols());
    std::iota(modelOrder.begin(), modelOrder.end(), 0);
    std::sort(modelOrder.begin(), modelOrder.end(), [&](int x, int y) {
        if (modelMass[x] != modelMass[y]) return modelMass[x] > modelMass[y];
        return x < y;
    });
    std::vector<int> selected;
    double selectedMass = 0.0;
    for (int k : modelOrder) {
        if (selectedMass >= factorMassThreshold * totalModelMass) break;
        if (!(modelMass[k] > 0.0)) break;
        selected.push_back(k);
        selectedMass += modelMass[k];
    }
    if (selected.empty()) throw std::runtime_error("No nonempty target factors selected");
    std::vector<int> selectedIndex(model.cols(), -1);
    std::vector<std::string> selectedTopics;
    RowMatrix selectedModel(model.rows(), selected.size());
    for (size_t j = 0; j < selected.size(); ++j) {
        selectedIndex[selected[j]] = static_cast<int>(j);
        selectedTopics.push_back(topics[selected[j]]);
        selectedModel.col(j) = model.col(selected[j]);
    }
    notice("Selected %zu of %d target factors covering %.6f of model mass",
        selected.size(), model.cols(), selectedMass / totalModelMass);
    const TargetMoments target = readTargetMoments(resultsPath, selectedTopics,
        selected.size() < static_cast<size_t>(model.cols()), loadingPruneThreshold);
    notice("Read %lld target units over selected factors (%lld skipped)",
        static_cast<long long>(target.units), static_cast<long long>(target.skippedUnits));
    notice("Target loading pruning at %.3g: mean %.2f active factors, mean %.3g and max %.3g discarded mass per retained unit",
        loadingPruneThreshold,
        static_cast<double>(target.activeLoadings) / target.units,
        target.discardedMass / target.units, target.maxDiscardedMass);

    std::ofstream summary(outPrefix + ".target_factors.tsv");
    if (!summary) throw std::runtime_error("Cannot open target factor output");
    summary << "Rank\tFactor\tModelMass\tModelMassFraction"
        "\tCumulativeModelMassFraction\tIncluded\tAbundance\tEffectiveUnits\n"
        << std::setprecision(6);
    double cumulativeMass = 0.0;
    for (size_t rank = 0; rank < modelOrder.size(); ++rank) {
        const int k = modelOrder[rank];
        const int j = selectedIndex[k];
        cumulativeMass += modelMass[k];
        summary << rank + 1 << '\t' << topics[k] << '\t' << modelMass[k]
            << '\t' << modelMass[k] / totalModelMass << '\t'
            << cumulativeMass / totalModelMass << '\t' << (j >= 0 ? 1 : 0);
        if (j >= 0) {
            const double effective = target.G(j, j) > 0.0
                ? target.units * target.a[j] * target.a[j] / target.G(j, j) : 0.0;
            summary << '\t' << target.a[j] << '\t' << effective << '\n';
        } else {
            summary << "\tNA\tNA\n";
        }
    }

    std::ofstream factorOutput(outPrefix + ".factor_scores.tsv");
    if (!factorOutput) throw std::runtime_error("Cannot open factor score output");
    factorOutput << "Reference\tTau\tFactor\tLowerBound\tUpperBound\tGap"
        "\tMeanLoss\tSecondLoss\tTrace\tIterations\tStatus\n";
    factorOutput << std::setprecision(6);
    std::vector<ScoreRow> scores;
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
    std::vector<ReferenceResult> panelResults(references.size());
    std::atomic<size_t> unfinishedPanels(references.size());
    std::mutex progressMutex;
    size_t completedPanels = 0;
    const size_t factorParallelCutoff = static_cast<size_t>(threads / 2);
    notice("Screening %zu reference panels with %d total threads",
        references.size(), threads);
    tbb::parallel_for(0, static_cast<int>(references.size()), [&](int panel) {
        const std::string& path = references[panel];
        const std::string& id = ids[panel];
        ReferenceResult& output = panelResults[panel];
        try {
            std::ostringstream factorRows;
            factorRows << std::setprecision(6);
            RowMatrix reference;
            std::vector<std::string> refFeatures, headers;
            readReferenceMatrix(path, reference, refFeatures, headers);
            checkMatrix(reference, path);
            int shared = 0;
            const ProjectedTarget projected = prepareReference(selectedModel, modelFeatures,
                target, reference, refFeatures, minSharedFeatures, shared);
            const int K = selectedModel.cols(), C = reference.cols();
            std::vector<std::vector<FactorSolve>> results(K);
            auto solveTopic = [&](int k) {
                if (!(target.a[k] > 0.0)) return;
                const FactorObjective objective(projected.B, projected.means[k],
                    projected.seconds[k], projected.meanScale, projected.secondScale);
                Vector weights = Vector::Zero(objective.count());
                weights.head(C).setConstant(1.0 / C);
                results[k].reserve(tauValues.size());
                for (double tau : tauValues) {
                    FactorSolve fit = solveFactor(objective, C, tau,
                        weights, maxIter, tolerance);
                    weights = fit.weights;
                    results[k].push_back(std::move(fit));
                }
            };
            for (int k = 0; k < K; ++k) {
                if (unfinishedPanels.load(std::memory_order_relaxed)
                        <= factorParallelCutoff) {
                    tbb::parallel_for(k, K, solveTopic);
                    break;
                }
                solveTopic(k);
            }
            for (size_t t = 0; t < tauValues.size(); ++t) {
                ScoreRow row;
                row.reference = id;
                row.path = path;
                row.status = "converged";
                row.tau = tauValues[t];
                row.maxGap = 0.0;
                row.sharedFeatures = shared;
                row.referenceFeatures = reference.rows();
                row.labels = C;
                std::vector<double> factorLower, factorUpper;
                factorLower.reserve(K);
                factorUpper.reserve(K);
                for (int k = 0; k < K; ++k) {
                    if (results[k].empty()) continue;
                    const auto& fit = results[k][t];
                    factorRows << id << '\t' << row.tau << '\t' << selectedTopics[k]
                        << '\t' << fit.lower << '\t' << fit.upper << '\t'
                        << fit.gap << '\t' << fit.meanLoss << '\t'
                        << fit.secondLoss << '\t' << fit.trace << '\t'
                        << fit.iterations << '\t'
                        << (fit.converged ? "converged" : "max_iter") << '\n';
                    factorLower.push_back(fit.lower);
                    factorUpper.push_back(fit.upper);
                    if (row.worstFactor < 0 || fit.upper > row.upper[0]) {
                        row.upper[0] = fit.upper;
                        row.worstFactor = k;
                    }
                    row.maxGap = std::max(row.maxGap, fit.gap);
                    row.iterations = std::max(row.iterations, fit.iterations);
                    if (!fit.converged) row.status = "max_iter";
                }
                row.selectedFactors = static_cast<int>(factorUpper.size());
                constexpr std::array<double, 4> fractions = {1.0, 0.9, 0.75, 0.5};
                for (size_t m = 0; m < fractions.size(); ++m) {
                    row.lower[m] = percentile(factorLower, fractions[m]);
                    row.upper[m] = percentile(factorUpper, fractions[m]);
                }
                output.scores.push_back(std::move(row));
            }
            output.factorRows = factorRows.str();
            output.sharedFeatures = shared;
            output.labels = C;
        } catch (const std::exception& ex) {
            output.error = ex.what();
            output.scores.clear();
        }
        unfinishedPanels.fetch_sub(1, std::memory_order_relaxed);
        if (verbose > 0) {
            std::lock_guard<std::mutex> lock(progressMutex);
            ++completedPanels;
            if (completedPanels % static_cast<size_t>(verbose) == 0
                    || completedPanels == references.size()) {
                notice("Screened %zu/%zu reference panels",
                    completedPanels, references.size());
            }
        }
    });
    for (size_t panel = 0; panel < panelResults.size(); ++panel) {
        const auto& output = panelResults[panel];
        if (output.error.empty()) {
            factorOutput << output.factorRows;
        } else {
            warning("Skipping reference %s: %s", ids[panel].c_str(), output.error.c_str());
        }
        scores.insert(scores.end(), output.scores.begin(), output.scores.end());
    }
    std::sort(scores.begin(), scores.end(), [](const ScoreRow& x, const ScoreRow& y) {
        if (x.tau != y.tau) return x.tau < y.tau;
        if (std::isnan(x.upper[0])) return false;
        if (std::isnan(y.upper[0])) return true;
        if (x.upper[0] != y.upper[0]) return x.upper[0] < y.upper[0];
        return x.reference < y.reference;
    });
    for (size_t first = 0; first < scores.size();) {
        size_t last = first + 1;
        while (last < scores.size() && scores[last].tau == scores[first].tau) ++last;
        std::vector<ScoreRow*> group;
        for (size_t i = first; i < last; ++i) group.push_back(&scores[i]);
        for (size_t metric = 0; metric < 4; ++metric) {
            std::sort(group.begin(), group.end(), [metric](const ScoreRow* x, const ScoreRow* y) {
                if (std::isnan(x->upper[metric])) return false;
                if (std::isnan(y->upper[metric])) return true;
                if (x->upper[metric] != y->upper[metric]) {
                    return x->upper[metric] < y->upper[metric];
                }
                return x->reference < y->reference;
            });
            int rank = 0;
            for (ScoreRow* row : group) {
                if (std::isfinite(row->upper[metric])) row->rank[metric] = ++rank;
            }
        }
        first = last;
    }
    std::ofstream scoreOutput(outPrefix + ".reference_scores.tsv");
    if (!scoreOutput) throw std::runtime_error("Cannot open reference score output");
    scoreOutput << "Tau\tRank\tReference\tLowerBound\tUpperBound\tMaxGap"
        "\tP90Rank\tP90LowerBound\tP90UpperBound"
        "\tP75Rank\tP75LowerBound\tP75UpperBound"
        "\tP50Rank\tP50LowerBound\tP50UpperBound"
        "\tWorstFactor\tSharedFeatures\tTargetCoverage\tReferenceFeatures"
        "\tLabels\tSelectedFactors\tTargetUnits\tSkippedUnits"
        "\tMaxIterations\tStatus\tPath\n"
        << std::setprecision(6);
    for (const auto& row : scores) {
        scoreOutput << row.tau << '\t' << row.rank[0]
            << '\t' << row.reference << '\t' << row.lower[0] << '\t'
            << row.upper[0] << '\t' << row.maxGap;
        for (size_t metric = 1; metric < 4; ++metric) {
            scoreOutput << '\t' << row.rank[metric] << '\t' << row.lower[metric]
                << '\t' << row.upper[metric];
        }
        scoreOutput << '\t'
            << (row.worstFactor >= 0 ? selectedTopics[row.worstFactor] : "NA") << '\t'
            << row.sharedFeatures << '\t'
            << static_cast<double>(row.sharedFeatures) / model.rows() << '\t'
            << row.referenceFeatures << '\t' << row.labels << '\t'
            << row.selectedFactors << '\t' << target.units << '\t'
            << target.skippedUnits << '\t'
            << row.iterations << '\t' << row.status << '\t' << row.path << '\n';
    }
    notice("Reference screen written to %s.reference_scores.tsv", outPrefix.c_str());

    return 0;
}
