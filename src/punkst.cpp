#include "punkst.h"

int32_t test(int32_t argc, char** argv);
int32_t cmdPts2TilesTsv(int32_t argc, char** argv);
int32_t cmdPts2TilesBinary(int32_t argc, char** argv);
int32_t cmdTiles2HexTxt(int32_t argc, char** argv);
int32_t cmdFASTQscribble(int32_t argc, char** argv);
int32_t cmdTsvDrawByColumn(int32_t argc, char** argv);
int32_t cmdTsvAnnoSb(int32_t argc, char** argv);
int32_t cmdImgNucleiMask(int32_t argc, char** argv);
int32_t cmdImgNucleiCenter(int32_t argc, char** argv);
int32_t cmdLDA4Hex(int argc, char** argv);
int32_t cmdPixelDecode(int32_t argc, char** argv);

int32_t main(int32_t argc, char** argv) {

  commandList cl;
  BEGIN_LONG_COMMANDS(longCommandlines)
    LONG_COMMAND_GROUP("Random Functions for Spatial Transcriptomics", NULL)
    LONG_COMMAND("test", &test, "Test")
    LONG_COMMAND("pts2tiles", &cmdPts2TilesTsv, "Assign points to tiles")
    LONG_COMMAND("pts2tiles-binary", &cmdPts2TilesBinary, "Assign points to tiles in binary format")
    LONG_COMMAND("tiles2hex", &cmdTiles2HexTxt, "Convert points in tiles to hexagons")
    LONG_COMMAND("lda4hex", &cmdLDA4Hex, "Train LDA model")
    LONG_COMMAND("pixel-decode", &cmdPixelDecode, "FICTURE pixel level decoding")
    LONG_COMMAND("scribble-parse", &cmdFASTQscribble, "Parse SCRIBBLE reads")
    LONG_COMMAND("draw-by-column", &cmdTsvDrawByColumn, "Given a TSV file and RGB assigned to columns, draw a PNG file")
    LONG_COMMAND("annotate-sb-tsv", &cmdTsvAnnoSb, "Annotate spatial locations by matching barcodes with 1 mismatch correction")
    LONG_COMMAND("nuclei-mask", &cmdImgNucleiMask, "Generate nuclei mask from unspliced and spliced read images")
    LONG_COMMAND("nuclei-center", &cmdImgNucleiCenter, "Find nuclei centers and annotate transcripts with distance to the nearest nuclei")

  END_LONG_COMMANDS();

  cl.Add(new longCommands("Available Commands", longCommandlines));

  if ( argc < 2 ) {
    fprintf(stderr, " Licensed under the Apache License v2.0 http://www.apache.org/licenses/\n\n");
    fprintf(stderr, "To run a specific command      : %s [command] [options]\n",argv[0]);
    fprintf(stderr, "For detailed instructions, run : %s --help\n",argv[0]);
    cl.Status();
    return 1;
  }
  else {
    if ( strcmp(argv[1],"--help") == 0 ) {
      cl.HelpMessage();
    }
    else {
      return cl.Read(argc, argv);
    }
  }
  return 0;
}
