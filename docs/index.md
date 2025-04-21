# punkst

punkst collects tools for analyzing high resolution spatial transcriptomics data.

## Pixel level factor analysis

The punkst toolkit provides a pipeline for efficient pixel level factor analysis, which achieves the same result as FICTURE (2024) with improved efficiency.

### Workflow Overview

Check [the quick start page](examples/basic.md) for help on generating the full workflow using Makefile or running each command step by step.

1. [pts2tiles](modules/pts2tiles.md): Group pixels to tiles for faster processing
2. [tiles2hex](modules/tiles2hex.md): Group pixels into non-overlapping hexagons
3. [lda4hex](modules/lda4hex.md): Run factorization on the hexagon data
4. [pixel-decode](modules/pixel-decode.md): Annotate each pixel with the top factors and their probabilities
5. [Visualization](modules/visualization.md): Create a high resolution visualization of the results

All analyses are parallelized and some steps need to write temporary files. Choose the number of threads and the temporary directory (must be empty or can be created) to match your system capabilities.

Check the [Modules](modules/index.md) section for detailed documentation of each command.
