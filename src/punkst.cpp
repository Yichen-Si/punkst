#include "punkst.h"

int32_t test(int32_t argc, char** argv);
int32_t cmdPts2TilesTsv(int32_t argc, char** argv);
int32_t cmdPts2TilesBinary(int32_t argc, char** argv);
int32_t cmdTiles2HexTxt(int32_t argc, char** argv);
int32_t cmdLDA4Hex(int argc, char** argv);
int32_t cmdPixelDecode(int32_t argc, char** argv);
int32_t cmdDrawPixelFactors(int32_t argc, char** argv);
int32_t cmdTiles2FeatureCooccurrence(int32_t argc, char** argv);
int32_t cmdQ2Markers(int32_t argc, char** argv);
int32_t cmdConvertDGE(int argc, char** argv);

int32_t main(int32_t argc, char** argv) {

    CommandList cl;
    cl.add_command("test", "Test command", "Test", test)
        .add_command("pts2tiles", "Assign points to tiles", "Assign points to tiles", cmdPts2TilesTsv)
        .add_command("pts2tiles-binary", "Assign points to tiles (binary)", "Assign points to tiles in binary format", cmdPts2TilesBinary)
        .add_command("tiles2hex", "Convert tiles to hexagons", "Convert points in tiles to hexagons", cmdTiles2HexTxt)
        .add_command("lda4hex", "Train LDA model", "Train LDA model", cmdLDA4Hex)
        .add_command("pixel-decode", "Decode pixel-level data", "Decoding pixel-level data", cmdPixelDecode)
        .add_command("draw-pixel-factors", "Draw pixel factors", "Draw pixel level factors", cmdDrawPixelFactors)
        .add_command("cooccurrence", "Compute feature co-occurrence", "Compute feature co-occurrence within a given radius", cmdTiles2FeatureCooccurrence)
        .add_command("coloc2markers", "Select markers from co-occurrence matrix", "Select markers from co-occurrence matrix", cmdQ2Markers)
        .add_command("convert-dge", "Convert DGE files to a TSV", "Convert DGE files to a sginel TSV (to prepare Visium HD input)", cmdConvertDGE);

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
