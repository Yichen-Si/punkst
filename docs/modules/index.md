# Punkst Modules

Punkst provides several command-line tools for analyzing high resolution spatial (transcriptomics) data. Each module can be used individually or as part of a pipeline.

## Available Modules

- [pts2tiles](pts2tiles.md): Groups pixels to tiles for faster processing
- [tiles2hex](tiles2hex.md): Groups pixels into non-overlapping hexagons for spot level analysis
- [lda4hex](lda4hex.md): Runs LDA on the spot level data
- [pixel-decode](pixel-decode.md): Annotates each pixel with the top factors and their probabilities
- [Visualization](visualization.md): Visualizes the pixel level analysis results

- [cooccurrence](coexp.md): Computes gene co-occurrence and/or extract marker genes from the co-occurrence matrix.

## Input Data Format

The input is a tsv file with the following columns: X, Y, feature, count. Whether the file contains headers or other columns is not relevant, as long as the above four columns are present.

- X, Y coordinates can be either integer or float vlaues. (If your coordinates are integers and you would like to keep the original coordinates in the pixel level inference output, set `--coords-are-int` in `punkst pixel-decode`). The coordinates can be in arbitrary units, just make sure all scale/size related parameters you later provide should be in the same unit.

- "feature" can be a string or a nonnegative integer corresponding to the index in a feature list.

- "count" is a nonnegative integer. You could apply gene-specific non-negative real valued weights to the count later in analysis.
