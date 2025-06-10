# pixel-decode

## Overview

`pixel-decode` takes a trained LDA model and tiled pixel-level data to annotate each pixel with the top factors and their probabilities. This module enables spatial mapping of gene expression patterns at single-pixel resolution.

```bash
punkst pixel-decode --model ${path}/hex_12.model.tsv \
--in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
--temp-dir ${tmpdir} --out-pref ${path}/pixel.decode \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
--hex-grid-dist 12 --n-moves 2 \
--pixel-res 0.5 --threads ${threads} --seed 1 --output-original
```

The pixel-level inference result (in this case `${path}/pixel.decode.tsv`) contains the coordinates and the inferred top factors and their posterior probabilities for each pixel. The module also creates a pseudobulk file (`${path}/pixel.decode.pseudobulk.tsv`) where each row is a gene and each column is a factor.

## Required Parameters

`--in-tsv` - Specifies the tiled data created by `pts2tiles`.

`--in-index` - Specifies the index file created by `pts2tiles`.

`--icol-x`, `--icol-y` - Specify the columns with X and Y coordinates (0-based).

`--icol-feature` - Specifies the column index for feature (0-based).

`--icol-val` - Specifies the column index for count/value (0-based).

`--model` - Specifies the model file where the first column contains feature names and the subsequent columns contain the parameters for each factor. The format should match that created by `lda4hex`.

`--temp-dir` - Specifies the directory to store temporary files.

**Output specification** - One of these must be provided:
`--out-pref` - Specifies the output prefix for all output files.

`--out` - (Deprecated, for backward compatibility) Specifies the output file.

**Hexagon grid parameters** - One of these must be provided:
`--hex-size` - Specifies the size (side length) of the hexagons for initializing anchors.

`--hex-grid-dist` - Specifies center-to-center distance in the axial coordinate system used to place anchors. Equals `hex-size * sqrt(3)`.

**Anchor spacing parameters** - One of these must be provided:
`--anchor-dist` - Specifies the distance between adjacent anchors.

`--n-moves` - Specifies the number of sliding moves in each axis to generate the anchors. If `--n-moves` is `n`, `anchor-dist` equals `hex-grid-dist` / `n`.

## Optional Parameters

### Input Parameters

`--coords-are-int` - If set, indicates that the coordinates are integers; otherwise, they are treated as floating point values.

`--feature-is-index` - If set, the values in `--icol-feature` are interpreted as feature indices. Otherwise, they are expected to be feature names.

`--feature-weights` - Specifies a file to weight each feature. The first column should contain the feature names, and the second column should contain the weights.

`--default-weight` - Specifies the default weight for features not present in the weights file (only if `--feature-weights` is specified). Default: 0.

`--anchor` - Specifies a file containing anchor points to use in addition to evenly spaced lattice points.

### Data Annotation Parameters

`--ext-col-ints` - Additional integer columns to carry over to the output file. Format: "idx1:name1 idx2:name2 ..." where 'idx' are 0-based column indices.

`--ext-col-floats` - Additional float columns to carry over to the output file. Format: "idx1:name1 idx2:name2 ..." where 'idx' are 0-based column indices.

`--ext-col-strs` - Additional string columns to carry over to the output file. Format: "idx1:name1:len1 idx2:name2:len2 ..." where 'idx' are 0-based column indices and 'len' are maximum lengths of strings.

### Processing Parameters

`--pixel-res` - Resolution for the analysis, in the same unit as the input coordinates. Default: 1 (each pixel treated independently). Setting the resolution equivalent to 0.5-1μm is recommended, but it could be smaller if your data is very dense.

`--radius` - Specifies the radius within which to search for anchors. Default: `anchor-dist * 1.2`.

`--min-init-count` - Minimum total count within the hexagon around an anchor for it to be included. Filters out regions outside tissues with sparse noise. Default: 10.

`--mean-change-tol` - Tolerance for convergence in terms of the mean absolute change in the topic proportions of a document. Default: 1e-3.

`--threads` - Number of threads to use for parallel processing. Default: 1.

`--seed` - Random seed for reproducibility. If not set or ≤0, a random seed will be generated.

### Output Parameters

`--output-original` - If set, the original data including the feature names and counts will be included in the output. If `pixel-res` is not 1 and `--output-original` is not set, the output contains results per collapsed pixel.

`--use-ticket-system` - If set, the order of pixels in the output file is deterministic across runs (though not necessarily the same as the input order). May incur a small performance penalty.

`--top-k` - Number of top factors to include in the output. Default: 3.

`--output-coord-digits` - Number of decimal digits to output for coordinates (only used if input coordinates are float or `--output-original` is not set). Default: 4.

`--output-prob-digits` - Number of decimal digits to output for probabilities. Default: 4.

`--verbose` - Increase verbosity of output messages.

`--debug` - Enable debug mode for additional diagnostic information.

## Process multiple samples

If you want to project the same model onto multiple datasets/samples, you can use `--sample-list` to pass a tsv file containing all samples' information. (In this case, `--in-tsv` and `--in-index` are ignored.)
Note: all other parameters are shared across samples, so the input files should have the same structure.

If your multi-sample data are generated by `punkst multisample-prepare`, it has created a file named `*.persample_file_list.tsv`. You can just pass this file to `--sample-list` and optionally use `--out-pref` to specify an identifier (e.g., the model information) to add to each output file name.

If you created the input file for each sample manually, you can create a tsv file with at least three columns:
sample_id, path to the transcript file created by `pts2tiles`, path to the index file created by `pts2tiles`.

Optional fourth column: output prefix.

Optional fifth column: anchor file path.

If there are headers, all header lines should start with "#".

## Output Files

The following files are generated (with prefix specified by `--out-pref`):

`<prefix>.tsv` - Main output file with pixel-level factor assignments

`<prefix>.pseudobulk.tsv` - Gene-by-factor matrix showing feature distribution across topics

## Example Usage Scenarios

### Basic Usage
```bash
punkst pixel-decode --model model.tsv --in-tsv data.tsv --in-index data.index \
--temp-dir /tmp --out-pref results \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
--hex-grid-dist 12 --n-moves 2 --threads 8
```

### With Data Annotations
```bash
punkst pixel-decode --model model.tsv --in-tsv data.tsv --in-index data.index \
--temp-dir /tmp --out-pref results \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
--hex-grid-dist 12 --n-moves 2 --threads 8 \
--ext-col-ints 4:celltype 5:cluster --ext-col-strs 6:sample_id:20 --output-original
```

### With multiple input created by `punkst multisample-prepare`
```bash
punkst pixel-decode --model model.tsv --sample-list multi.persample_file_list.tsv \
--temp-dir /tmp --out-pref results \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
--hex-grid-dist 12 --n-moves 2 --threads 8
```
