# punkst

## Pixel level factor analysis

The punkst toolkit provides a pipeline for efficient pixel level factor analysis, which achieves the same result as[ FICTURE (2024)](https://doi.org/10.1038/s41592-024-02415-2) with improved efficiency. ([Previous python package](https://github.com/seqscope/ficture) is still functioning.)


Check [the quick start page](workflows/index.md) for help on generating the full workflow using Makefile or running each command step by step.

Check the [Input](input/index.md) for help on preparing your data. Current we have worked exampled for 10X Genomics Visium HD, Xenium, NanoString CosMx SMI, and Vizgen MERSCOPE data. We've also applied punkst to Seq-scope and Stereo-seq data, we are working on providing more information.

### FICTURE Workflow Overview

1. [pts2tiles](modules/pts2tiles.md): Group pixels to tiles for faster processing
2. [tiles2hex](modules/tiles2hex.md): Group pixels into non-overlapping hexagons
3. [lda4hex](modules/lda4hex.md): Run factorization on the hexagon data
4. [pixel-decode](modules/pixel-decode.md): Annotate each pixel with the top factors and their probabilities
5. [Visualization](modules/visualization.md): Create a high resolution visualization of the results

All analyses are parallelized and some steps need to write temporary files. Choose the number of threads and the temporary directory (must be empty or can be created) to match your system capabilities.

Check the [Modules](modules/index.md) for detailed documentation of each command and additional tools.
