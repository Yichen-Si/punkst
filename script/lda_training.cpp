#include "punkst.h"
#include "lda.hpp"
#include "dataunits.hpp"

class LDA4Hex {

public:

    LDA4Hex(const std::string& metaFile, int32_t layer = 0) : reader(metaFile), layer(layer) {
        if (layer >= reader.getNLayer()) {
            error("layer %d is out of range", layer);
        }
        if (reader.nFeatures <= 0) {
            error("n_features is missing in the metadata");
        }
        notice("Read metadata: %zu features, %d hexagons with side length %.2f", reader.nFeatures, reader.nUnits, reader.hexSize);
        weightFeatures = false;
        ntot = 0;
        minCountTrain = 0;
    }
    void setFeatureNames(const std::vector<std::string>& featureNames) {
        reader.setFeatureNames(featureNames);
    }
    void setFeatureNames(const std::string& nameFile) {
        reader.setFeatureNames(nameFile);
    }

    void readWeights(const std::string& weightFile, double defaultWeight = 1.0) {
        std::ifstream inWeight(weightFile);
        if (!inWeight) {
            error("Error opening weights file: %s", weightFile.c_str());
        }
        int32_t nweighted = 0, novlp = 0;
        weights.resize(reader.nFeatures);
        std::fill(weights.begin(), weights.end(), defaultWeight);
        std::unordered_map<std::string, uint32_t> featureDict;
        bool foundDict = reader.featureDict(featureDict);
        if (!foundDict) { // then the weight file must refer to features by their index as in the input file
            warning("Feature dictionary is not found in the input metadata, so we assume the weight file (--feature-weights) refers to features by their index as in the input file");
            std::string line;
            while (std::getline(inWeight, line)) {
                nweighted++;
                std::istringstream iss(line);
                uint32_t idx;
                double weight;
                if (!(iss >> idx >> weight)) {
                    error("Error reading weights file: %s", weightFile.c_str());
                }
                if (idx >= reader.nFeatures) {
                    warning("Input file contains %zu features, feature index %u in the weights file is out of range", reader.nFeatures, idx);
                    continue;
                }
                weights[idx] = weight;
                novlp++;
            }
        } else { // assume the weight file refers to features by their names
            std::string line;
            while (std::getline(inWeight, line)) {
                nweighted++;
                std::istringstream iss(line);
                std::string feature;
                double weight;
                if (!(iss >> feature >> weight)) {
                    error("Error reading weights file: %s", weightFile.c_str());
                }
                auto it = featureDict.find(feature);
                if (it != featureDict.end()) {
                    weights[it->second] = weight;
                    novlp++;
                } else {
                    warning("Feature %s not found in the input", feature.c_str());
                }
            }
        }
        notice("Read %d weights from file, %d features are overlapped with the input file", nweighted, novlp);
        if (novlp == 0) {
            error("No features in the weight file overlap with those found in the input file, check if the files are consistent.");
        }
        weightFeatures = true;
    }

    int32_t trainOnline(LatentDirichletAllocation& lda, const std::string& inFile, int32_t _bsize = 128, int32_t _minCountTrain = 0) {
        batchSize = _bsize;
        minCountTrain = _minCountTrain;
        ntot = 0;
        std::ifstream inFileStream(inFile);
        if (!inFileStream) {
            error("Error opening input file: %s", inFile.c_str());
        }
        bool fileopen = true;
        int nbatch = 0;
        while (fileopen) {
            fileopen = readMinibatch(inFileStream);
            if (minibatch.empty()) {
                break;
            }
            lda.partial_fit(minibatch);
            ntot += minibatch.size();
            nbatch++;
        }
        inFileStream.close();
        return ntot;
    }


    void writeModelToFile(LatentDirichletAllocation& lda, const std::string& outFile) {
        std::ofstream outFileStream(outFile);
        if (!outFileStream) {
            error("Error opening output file: %s for writing", outFile.c_str());
        }
        const MatrixXd& model = lda.get_model();
        int32_t nTopics = model.rows();
        int32_t nFeatures = model.cols();
        outFileStream << "Feature";
        for (int i = 0; i < nTopics; ++i) {
            outFileStream << "\t" << i;
        }
        outFileStream << "\n";
        outFileStream << std::fixed << std::setprecision(3);
        for (int i = 0; i < nFeatures; ++i) {
            outFileStream << reader.features[i];
            for (int j = 0; j < nTopics; ++j) {
                outFileStream << "\t" << model(j, i);
            }
            outFileStream << "\n";
        }
        outFileStream.close();
    }

    void fitAndWriteToFile(LatentDirichletAllocation& lda, const std::string& inFile, const std::string& outFile) {
        std::ifstream inFileStream(inFile);
        if (!inFileStream) {
            error("Error opening input file: %s", inFile.c_str());
        }
        std::ofstream outFileStream(outFile);
        if (!outFileStream) {
            error("Error opening output file: %s for writing", outFile.c_str());
        }
        int32_t nTopics = lda.get_n_topics();
        outFileStream << "x\ty";
        for (int i = 0; i < nTopics; ++i) {
            outFileStream << "\t" << i;
        }
        outFileStream << "\n";

        bool fileopen = true;
        while (fileopen) {
            std::vector<std::string> idens;
            fileopen = readMinibatch(inFileStream, idens);
            if (minibatch.empty()) {
                break;
            }
            Eigen::MatrixXd doc_topic = lda.transform(minibatch);
            for (int i = 0; i < doc_topic.rows(); ++i) {
                double sum = doc_topic.row(i).sum();
                if (sum > 0) {
                    doc_topic.row(i) /= sum;
                }
            }
            for (int i = 0; i < minibatch.size(); ++i) {
                outFileStream << idens[i] << std::fixed << std::setprecision(4);
                for (int j = 0; j < nTopics; ++j) {
                    outFileStream << "\t" << doc_topic(i, j);
                }
                outFileStream << "\n";
            }
        }
        inFileStream.close();
        outFileStream.close();
    }

    int32_t nUnits() const {
        return reader.nUnits;
    }
    int32_t nFeatures() const {
        return reader.nFeatures;
    }

private:

    HexReader reader;

    int32_t layer;
    int32_t ntot;
    int32_t minCountTrain;
    bool weightFeatures;
    std::vector<double> weights;

    std::vector<Document> minibatch;
    int32_t batchSize;

    bool readMinibatch(std::ifstream& inFileStream) {
        minibatch.clear();
        minibatch.reserve(batchSize);
        std::string line;
        int32_t nlocal = 0;
        while (nlocal < batchSize) {
            if (!std::getline(inFileStream, line)) {
                return false;
            }
            Document doc;
            int32_t ct = reader.parseLine(doc, line, layer);
            if (ct < 0) {
                error("Error parsing line %s", line.c_str());
            }
            if (ct < minCountTrain) {
                continue;
            }
            if (weightFeatures) {
                for (size_t i = 0; i < doc.ids.size(); ++i) {
                    doc.cnts[i] *= weights[doc.ids[i]];
                }
            }
            minibatch.push_back(std::move(doc));
            nlocal++;
        }
        return true;
    }

    bool readMinibatch(std::ifstream& inFileStream, std::vector<std::string>& idens) {
        minibatch.clear();
        minibatch.reserve(batchSize);
        std::string line;
        int32_t nlocal = 0;
        while (nlocal < batchSize) {
            if (!std::getline(inFileStream, line)) {
                return false;
            }
            Document doc;
            int32_t x, y;
            int32_t ct = reader.parseLine(doc, x, y, line, layer);
            if (ct < 0) {
                error("Error parsing the %d-th line", ntot);
            }
            std::stringstream ss;
            if (reader.hexSize <= 0) {
                ss << x << "\t" << y;
            } else {
                double dx, dy;
                reader.hexGrid.axial_to_cart(dx, dy, x, y);
                ss << std::fixed << std::setprecision(3) << dx << "\t" << dy;
            }
            idens.push_back(ss.str());
            if (weightFeatures) {
                for (size_t i = 0; i < doc.ids.size(); ++i) {
                    doc.cnts[i] *= weights[doc.ids[i]];
                }
            }
            minibatch.push_back(std::move(doc));
            nlocal++;
        }
        return true;
    }

};

int32_t cmdLDA4Hex(int argc, char** argv) {

    std::string inFile, metaFile, weightFile, outPrefix, nameFile;
    int32_t nTopics;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 128;
    int32_t verbose = 0;
    int32_t maxIter = 100;
    int32_t nThreads = 0;
    int32_t layer = 0;
    int32_t minCountTrain = 20;
    double kappa = 0.7, tau0 = 10.0;
    double alpha = -1., eta = -1.;
    double mDelta = -1;
    double defaultWeight = 1.;
    bool transform = false;

    paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-data", &inFile, "Input hex file")
        LONG_STRING_PARAM("in-meta", &metaFile, "Metadata file")
        LONG_STRING_PARAM("feature-weights", &weightFile, "Input weights file")
        LONG_STRING_PARAM("feature-names", &nameFile, "Feature names")
        LONG_INT_PARAM("n-topics", &nTopics, "Number of topics")
        LONG_INT_PARAM("layer", &layer, "Layer to use (0-based)")
        LONG_INT_PARAM("seed", &seed, "Random seed")
        LONG_INT_PARAM("threads", &nThreads, "Number of threads to use (default: 1)")
        LONG_INT_PARAM("minibatch-size", &batchSize, "Minibatch size (default: 128)")
        LONG_INT_PARAM("min-count-train", &minCountTrain, "Minimum total count for training (default: 20)")
        LONG_DOUBLE_PARAM("mean-change-tol", &mDelta, "Mean change of document-topic probability tolerance for convergence (default: 1e-3)")
        LONG_INT_PARAM("n-epochs", &nEpochs, "Number of epochs (default: 1)")
        LONG_DOUBLE_PARAM("default-weight", &defaultWeight, "Default weight for features not in the provided weight file (default: 1.0, set it to 0 to ignore features not in the weight file)")
        LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out-prefix", &outPrefix, "Output hex file")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_PARAM("transform", &transform, "Transform the input data to the LDA space after training")

    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    if (inFile.empty() || metaFile.empty() || outPrefix.empty()) {
        error("--in-data --in-meta and --out are required");
    }
    std::string outModel = outPrefix + ".model.tsv";
    std::ofstream outFileStream(outModel);
    if (!outFileStream) {
        error("Error opening output file: %s for writing", outModel.c_str());
    }
    outFileStream.close();
    if (nTopics <= 0) {
        error("Number of topics must be greater than 0");
    }

    if (seed <= 0) {
        seed = std::random_device{}();
    }
    if (mDelta <= 0) {
        mDelta = 0.002 / nTopics;
    }

    if (batchSize <= 0) {
        batchSize = 128;
        warning("Minibatch size must be greater than 0, using default value of 128");
    }
    if (nEpochs <= 0) {
        nEpochs = 1;
    }

    LDA4Hex lda4hex(metaFile, layer);
    if (!nameFile.empty()) {
        lda4hex.setFeatureNames(nameFile);
    }
    if (!weightFile.empty()) {
        lda4hex.readWeights(weightFile, defaultWeight);
    }

    LatentDirichletAllocation lda(nTopics, lda4hex.nFeatures(), seed, nThreads, verbose, alpha, eta, maxIter, mDelta, kappa, tau0, lda4hex.nUnits() * nEpochs);

    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        int32_t n = lda4hex.trainOnline(lda, inFile, batchSize, minCountTrain);
        notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
    }

    // write model matrix to file
    lda4hex.writeModelToFile(lda, outModel);
    notice("Model written to %s", outModel.c_str());

    if (!transform) {
        return 0;
    }
    // transform the input data to the LDA space
    std::string outFile = outPrefix + ".results.tsv";
    lda4hex.fitAndWriteToFile(lda, inFile, outFile);
    notice("Transformed data written to %s", outFile.c_str());

    return 0;
};
