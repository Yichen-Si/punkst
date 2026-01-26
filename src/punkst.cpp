#include "punkst.h"

int32_t test(int32_t argc, char** argv);
int32_t cmdPts2TilesTsv(int32_t argc, char** argv);
int32_t cmdPts2TilesBinary(int32_t argc, char** argv);
int32_t cmdTiles2HexTxt(int32_t argc, char** argv);
int32_t cmdTopicModelSVI(int argc, char** argv);
int32_t cmdLDATransform(int argc, char** argv);
int32_t cmdPixelDecode(int32_t argc, char** argv);
int32_t cmdDrawPixelFactors(int32_t argc, char** argv);
int32_t cmdDrawLowresFactors(int32_t argc, char** argv);
int32_t cmdTiles2FeatureCooccurrence(int32_t argc, char** argv);
int32_t cmdMergeCooccurrenceMtx(int32_t argc, char** argv);
int32_t cmdQ2Markers(int32_t argc, char** argv);
int32_t cmdConvertDGE(int argc, char** argv);
int32_t cmdConvert10xToHexTSV(int argc, char** argv);
int32_t cmdMultiSample(int32_t argc, char** argv);
int32_t cmdMergeUnits(int32_t argc, char** argv);
int32_t cmdConditionalTestPoisReg(int argc, char** argv);
int32_t cmdConditionalTestNbReg(int argc, char** argv);
int32_t cmdNmfPoisLog1p(int32_t argc, char** argv);
int32_t cmdNmfTransform(int32_t argc, char** argv);
int32_t cmdDrawPixelFeatures(int32_t argc, char** argv);
int32_t cmdFeatureVst(int32_t argc, char** argv);
int32_t cmdDeChisq(int argc, char** argv);
int32_t cmdDeconvPseudobulk(int argc, char** argv);
int32_t cmdPseudoBulk(int argc, char** argv);
int32_t cmdManipulateTiles(int32_t argc, char** argv);
int32_t cmdConditionalTest(int32_t argc, char** argv);

int32_t main(int32_t argc, char** argv) {

    CommandList cl;
    cl.add_command("test", "Test", test)
        .add_command("pts2tiles", "Assign points to tiles", cmdPts2TilesTsv)
        .add_command("pts2tiles-binary", "Assign points to tiles in binary format", cmdPts2TilesBinary)
        .add_command("tiles2hex", "Convert points in tiles to hexagons", cmdTiles2HexTxt)
        .add_command("lda4hex", "Train LDA model", cmdTopicModelSVI) // backward compatibility
        .add_command("topic-model", "Train LDA/HDP model", cmdTopicModelSVI)
        .add_command("lda-transform", "Transform data using fitted LDA model", cmdLDATransform)
        .add_command("pixel-decode", "Decoding pixel-level data", cmdPixelDecode)
        .add_command("draw-pixel-factors", "Draw pixel level factors", cmdDrawPixelFactors)
        .add_command("draw-lowres-factors", "Draw low-resolution factor map", cmdDrawLowresFactors)
        .add_command("cooccurrence", "Compute feature co-occurrence within a given radius", cmdTiles2FeatureCooccurrence)
        .add_command("merge-mtx", "Merge multiple co-occurrence matrices", cmdMergeCooccurrenceMtx)
        .add_command("coloc2markers", "Select markers from co-occurrence matrix", cmdQ2Markers)
        .add_command("convert-dge", "Convert DGE files to a sginel TSV (to prepare Visium HD input)", cmdConvertDGE)
        .add_command("convert-10X-SC", "Convert 10X Genomics single-cell DGE to customized unit level tsv files", cmdConvert10xToHexTSV)
        .add_command("multisample-prepare", "Process multisample data", cmdMultiSample)
        .add_command("merge-units", "Merge multiple single cell/hexagon files into a single file", cmdMergeUnits)
        .add_command("multi-conditional-de-pois", "Multi-sample cell type specific DE test by Poisson regression (use --link=log|log1p)", cmdConditionalTestPoisReg)
        .add_command("multi-conditional-de-nb", "Multi-sample cell type specific DE test by Poisson regression with over-dispersion (NB) (use --link=log|log1p)", cmdConditionalTestNbReg)
        .add_command("multi-conditional-de-pixel", "Multi-sample cell type specific DE test based on pixel level cell type inference", cmdConditionalTest)
        .add_command("nmf-pois-log1p", "Fit Poisson log1p NMF", cmdNmfPoisLog1p)
        .add_command("nmf-pois-log1p-transform", "Transform data using fitted Poisson log1p NMF model", cmdNmfTransform)
        .add_command("draw-pixel-features", "Draw pixel level features/genes", cmdDrawPixelFeatures)
        .add_command("feature-vst", "Compute feature variance stabilizing stats and select highly variable features", cmdFeatureVst)
        .add_command("de-chisq", "Differential expression using Chi-squared test", cmdDeChisq)
        .add_command("deconv-pseudobulk", "Deconvolve pseudobulk matrix using a confusion matrix", cmdDeconvPseudobulk)
        .add_command("pseudo-bulk", "Generate pseudo-bulk matrix", cmdPseudoBulk)
        .add_command("tile-op", "View/manipulate tiles", cmdManipulateTiles);

    if (argc < 2) {
        std::cerr << "Licensed under the CC BY-NC 4.0 https://creativecommons.org/licenses/by-nc/4.0/\n\n";
        std::cerr << "To run a specific command      : " << argv[0] << " [command] [options]\n";
        std::cerr << "For detailed instructions, run : " << argv[0] << " --help\n";
        cl.print_help();
        return 1;
    } else if (std::strcmp(argv[1], "--help") == 0) {
        cl.print_help();
        return 0;
    } else {
        return cl.parse(argc, argv);
    }

}
