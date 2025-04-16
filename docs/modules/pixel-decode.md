# pixel-decode

### Pixel level decoding

`pixel-decode` takes a model and the tiled pixel level data to annotate each pixel with the top factors and their probabilities.

```bash
punkst pixel-decode --model ${path}/hex.lda.model.tsv --in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index --temp-dir ${tmpdir} --out ${path}/pixel.decode.tsv --icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 --hex-grid-dist 1200 --n-moves 2 --min-init-count 20 --pixel-res 50 --threads ${threads} --seed 1 --output-original
```

The pixel level inference result (in this case `${path}/pixel.decode.tsv`) contains the coordinates and the inferred top factors and their posterior probabilities for each pixel. We also create a pseudobulk file (`${path}/pixel.decode.pseudobulk.tsv`) where each row is a gene and each column is a factor.

## Required Parameters

`--in-tsv` specifies the tiled data created by `pts2tiles`.

`--in-index` specifies the index file created by `pts2tiles`.

`--icol-x`, `--icol-y` specify the columns with X and Y coordinates (0-based).

`--icol-feature` specifies the column index for feature (0-based).

`--icol-val` specifies the column index for count/value (0-based).

`--model` specifies the model file where the first column contains feature names and the subsequent columns contain the parameters for each factor. The format should match that created by `lda4hex`.

`--out` specifies the output file.

`--temp-dir` specifies the directory to store temporary files.

One of `--hex-size` or `--hex-grid-dist`: `--hex-size` specifies the size (side length) of the hexagons for initializing anchors; `--hex-grid-dist` specifies center-to-center distance in the axial coordinate system used to place anchors. `hex-grid-dist` equals `hex-size * sqrt(3)`.

One of `--anchor-dist` or `--n-moves`: `--anchor-dist` specifies the distance between adjacent anchors; `--n-moves` specifies the number of sliding moves in each axis to generate the anchors. If `--n-moves` is `n`, `anchor-dist` equals `hex-grid-dist` / `n`.

## Optional Parameters

### Input Parameters

`--coords-are-int` if set, indicates that the coordinates are integers; otherwise, they are treated as floating point values.

`--feature-is-index` if set, the values in `--icol-feature` are interpreted as feature indices. Otherwise, they are expected to be feature names.

`--feature-weights` specifies a file to weight each feature. The first column should contain the feature names, and the second column should contain the weights.

`--default-weight` specifies the default weight for features not present in the weights file (only if `--feature-weights` is specified). Default is 0.

`--anchor` specifies a file containing anchor points to use in addition to evenly spaced lattice points.

### Processing Parameters

`--pixel-res` resolution for the analysis, in the same unit as the input coordinates. The default is `1` so each pixel is treated independently. Setting the resolution equivalent to $0.5\sim 1 \mu m$ is recommended, but it could be smaller if your data is very dense.

`--radius` specifies the radius within which to search for anchors. If not specified, it defaults to `anchor-dist * 1.2`.

`--min-init-count` specifies the minimum total count within the hexagon around an anchor for the anchor to be included. It will filter out regions outside tissues with sparse noise. Default is 10.

`--threads` specifies the number of threads to use. Default is 1.

`--seed` specifies the random seed to use for reproducibility. If not provided, a random seed will be generated.

### Output Parameters

`--output-original` if set, the original data including the feature names and counts will be included in the output. If `pixel-res` is not `1` and `--output-original` is not set, the output contains results per collapsed pixel.

`--use-ticket-system` if set, the order of pixels in the output file is deterministic so the same between runs (though not the same asthat in the input). This may incurr a small performance penalty.

`--top-k` specifies the number of top factors to include in the output. Default is 3.

`--output-coord-digits` specifies the number of decimal digits to output for coordinates (only used if input coordinates are float or `--output-original` is not set). Default is 4.

`--output-prob-digits` specifies the number of decimal digits to output for probabilities. Default is 4.
