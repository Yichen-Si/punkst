# Punkst Modules

punkst provides several command-line tools for analyzing high resolution spatial (transcriptomics) data.

## Available Modules

- [pts2tiles](pts2tiles.md): Group pixels to tiles for faster processing
- [tiles2hex](tiles2hex.md): Group pixels into non-overlapping hexagons for spot level analysis
- [topic-model](lda4hex.md): Run LDA on the spot level data
- [pixel-decode](pixel-decode.md): Annotate each pixel with the top factors and their probabilities
- [cooccurrence](coexp.md): Compute gene co-occurrence and/or extract marker genes from the co-occurrence matrix
- [visualization](visualization.md): Visualize the pixel level analysis results
- [DE tests](de.md): Differential expression tests
- [tile-op](tileop.md): View and manipulate high resolution spatial inference output (merge, annotate, denoise, bin, etc.)
<!-- - [nmf-pois-log1p](poisnmf.md): Fits a Poisson NMF model with a log(1+x) link. -->

## Input Data Format

The input is a tsv file with the following columns: X, Y, feature, count. Whether the file contains headers or other columns is not relevant, as long as the above four columns are present.

- X, Y coordinates can be either integer or float vlaues. (If your coordinates are integers and you would like to keep the original coordinates in the pixel level inference output, set `--coords-are-int` in `punkst pixel-decode`). The coordinates can be in arbitrary units, just make sure all scale/size related parameters you later provide are in the same unit.

- "feature" column contains the gene names (strings) or a nonnegative integer corresponding to the index in a feature list.

- "count" is a nonnegative value for transcript counts (it does not have to be integers). If your data are from imaging platforms where each record is a single transcript, you might need to add this column manually with all `1`.
