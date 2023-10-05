#include "punkst.h"
#include "layout.h"

int32_t test(int32_t argc, char** argv) {

  std::string flayout, fmanifest, input, output;
  uint32_t precision;
  double scale;

  // Parse input parameters
  paramList pl;

  BEGIN_LONG_PARAMS(longParameters)
    LONG_PARAM_GROUP("Input options", NULL)
    LONG_STRING_PARAM("layout", &flayout, "Layout file")
    LONG_STRING_PARAM("manifest", &fmanifest, "")
    LONG_STRING_PARAM("input", &input, "")
    LONG_PARAM_GROUP("Output Options", NULL)
    LONG_STRING_PARAM("output", &output, "Output file")
    LONG_DOUBLE_PARAM("scale", &scale, "")
    LONG_INT_PARAM("precision", &precision, "")
  END_LONG_PARAMS();

  pl.Add(new longParams("Available Options", longParameters));
  pl.Read(argc, argv);
  pl.Status();

  // // check input file sanity
  // if ( manifest.empty() || layout.empty() || input.empty() || output.empty() )
  //   error("--input --output are required but at least one is missing");

  notice("Process Started");

  SpatialLayout layoutobj(flayout.c_str(), fmanifest.c_str(), 0, 1, 1, 1, 1);


  notice("Analysis Finished");

  return 0;
}
