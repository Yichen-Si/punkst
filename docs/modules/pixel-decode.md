# pixel-decode

## Overview

`pixel-decode` projects a trained factor model onto tiled pixel-level data and annotates each pixel (or collapsed pixel bin) with the top factors and their probabilities.

It supports:

- 2D decoding on overlapping shifted hex-anchor lattices
- thin 3D decoding, which keeps the shifted hex-anchor construction and adds a finite set of z levels
- standard 3D decoding, which uses a body-centered cubic (BCC) anchor lattice directly in the input 3D coordinate system

In all cases, the input remains the x-y tiled output created by `pts2tiles`.
<!-- The default decoder is `slda`; `--algo nmf` enables the EM-NMF decoder. -->

```bash
punkst pixel-decode --model ${path}/hex_12.model.tsv \
--in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
--temp-dir ${tmpdir} --out-pref ${path}/pixel --output-binary \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
--hex-grid-dist 12 --n-moves 2 \
--pixel-res 0.5 --threads ${threads} --seed 1
```

The main inference result in this example is written to `${path}/pixel.bin` together with `${path}/pixel.index`. The output are organized in regular square tiles for efficient downstream analysis (see [tile-op](tileop.md)).

## Required Parameters

`--in-tsv` - Specifies the tiled data created by `pts2tiles`.

`--in-index` - Specifies the index file created by `pts2tiles`.

`--icol-x`, `--icol-y` - Specify the columns with X and Y coordinates (0-based).

`--icol-feature` - Specifies the column index for feature (0-based).

`--icol-val` - Specifies the column index for count/value (0-based).

`--model` - Specifies the model file where the first column contains feature names and the subsequent columns contain the parameters for each factor. The format should match that created by `topic-model`.

`--temp-dir` - Specifies the directory used for temporary files. Required unless `--in-memory` is set.

**Output specification** - One of these must be provided:
`--out-pref` - Specifies the output prefix for all output files.

`--out` - (Deprecated, for backward compatibility) Specifies the output file.

**Hexagon grid parameters** for 2D and thin 3D, one of these must be provided:

`--hex-size` - Specifies the size (side length) of the hexagons for initializing anchors.

`--hex-grid-dist` - Specifies center-to-center distance in the axial coordinate system used to place anchors. Equals `hex-size * sqrt(3)`.

**Anchor spacing parameters**
`--anchor-dist` - Specifies the distance between adjacent anchors. Required for standard 3D. In 2D and thin 3D, it can be provided directly instead of `--n-moves`.

`--n-moves` - (For 2D and thin-3D) Specifies the number of sliding moves in each axis to generate the anchors. If `--n-moves` is `n`, `anchor-dist` equals `hex-grid-dist` / `n`.

## Optional Parameters

### Input Parameters

`--coords-are-int` - If set, indicates that the coordinates are integers; otherwise, they are treated as floating point values.

`--icol-z` - Column index for the z coordinate (0-based). Requires either
`--thin-3D` or `--standard-3D`.

`--thin-3D` - Activates the thin 3D path.

`--standard-3D` - Activates the standard 3D BCC path.

`--feature-is-index` - If set, the values in `--icol-feature` are interpreted as feature indices. Otherwise, they are expected to be feature names.

`--feature-weights` - Specifies a file to weight each feature. The first column should contain the feature names, and the second column should contain the weights.

`--default-weight` - Specifies the default weight for features not present in the weights file (only if `--feature-weights` is specified). Default: 0.

`--anchor` - Specifies a file containing fixed anchor points. This is currently
only supported for 2D `slda`. In 3D, fixed anchors are ignored. In `nmf`,
externally loaded anchors are not currently used.

`--sample-list` - Runs the same model and settings on multiple datasets listed in one TSV file. See "Process multiple samples" below.

`--in-memory` - Keeps boundary buffers in memory instead of writing temporary buffer files. If set, `--temp-dir` is not required.

### Algorithm Parameters

<!-- `--algo` - Decoding algorithm: `slda` (default) or `nmf`. -->

`--max-iter` - Maximum number of outer iterations. Default: 100.

`--mean-change-tol` - Convergence tolerance for the outer iterations. Default: `1e-3`.

`--background-model` - Background profile file. If provided, background probabilities are modeled explicitly.

`--bg-fraction-prior-a0`, `--bg-fraction-prior-b0` - Beta prior hyperparameters for the background fraction in `slda` mode.

<!-- `--bg-fraction` - Background fraction used by `nmf` mode.

`--model-bin` - Optional binary model used by the EM-NMF MLR path.

`--size-factor` - Size factor used for per-anchor EM updates in `nmf` mode.

`--exact` - Use exact Poisson updates in `nmf` mode.

`--max-iter-inner`, `--tol-inner` - Inner MLE solver controls for `nmf` mode.

`--weight-thres-anchor` - Minimum total weight for an anchor to be kept in `nmf` mode.

`--ridge` - Ridge stabilization parameter for the `nmf` inner solver. -->

### Data Annotation Parameters

`--ext-col-ints` - Additional integer columns to carry over to the output file. Format: "idx1:name1 idx2:name2 ..." where 'idx' are 0-based column indices. Example: `--ext-col-ints 4:celltype 5:cluster`.

`--ext-col-floats` - Additional float columns to carry over to the output file. Format: "idx1:name1 idx2:name2 ..." where 'idx' are 0-based column indices. Example: `--ext-col-floats 6:quality`.

`--ext-col-strs` - Additional string columns to carry over to the output file. Format: "idx1:name1:len1 idx2:name2:len2 ..." where 'idx' are 0-based column indices and 'len' are maximum lengths of strings. Example: `--ext-col-strs 7:sample_id:20 8:batch:10`.

### Processing Parameters

`--pixel-res` - Resolution for the analysis, in the same unit as the input coordinates. Default: 1 (each pixel treated independently). Setting the resolution equivalent to 0.5-1μm is recommended, but it could be smaller if your data is very dense. For Visium HD (where the pixel size is 2μm), use `--pixel-res 2`.

`--pixel-res-z` - Resolution for aggregating pixels in the z dimension. Used
only in 3D mode. Default: 1.

`--radius` - Support radius used in pixel-to-anchor weighting. In 2D and thin
3D it also controls the anchor search neighborhood. Default: `anchor-dist *
1.2` in 2D. In thin 3D, a default is derived from the x-y anchor spacing and
z-level spacing. In standard 3D, it does not change the fixed BCC neighbor
stencil, but it does control the weight decay scale and contributes to the
x-y padding.

`--half-life-dist` - Ratio `h` in `(0, 1)` such that an anchor at distance
`h * radius` receives weight `0.5`. Default: `0.7`. The implemented weighting
rule is `w(d) = clamp(1 - (d / radius)^nu, 0.05, 0.95)` with
`nu = log(0.5) / log(h)`.

`--min-init-count` - Minimum total count within the hexagon around an anchor for it to be included. Filters out regions outside tissues with sparse noise. Default: 10.

`--zmin`, `--zmax` - z range for 3D mode. Thin 3D requires both values.
Standard 3D accepts them, but only uses them when
`--ignore-outside-zrange` is set.

`--thin-3d-z-levels` - Explicit z coordinates for thin-3D anchor levels.

`--thin-3d-n-z-levels` - Number of evenly spaced thin-3D anchor levels to
generate between `zmin` and `zmax`.

`--ignore-outside-zrange` - Drop observations outside `[zmin, zmax]` in 3D mode.

`--n-init-anchor-per-pix` - Thin-3D-only initialization parameter controlling
how many nearby z levels each point contributes to when anchors are initialized.

`--threads` - Number of threads to use for parallel processing. Default: 1.

`--seed` - Random seed for reproducibility. If not set or ≤0, a random seed will be generated.

### Output Parameters

`--output-binary` - Writes the main output as `<prefix>.bin` plus `<prefix>.index`. This cannot be combined with `--output-original`.

`--output-original` - Writes the main output as text and includes the original feature names and counts together with the factor results.

`--output-anchors` - Writes anchor-level top-factor assignments to `<prefix>.anchors.tsv`.

`--output-bg-prob-expand` - In text mode, writes one line per feature per pixel to include background probabilities. Ignored if `--output-original` is set.

`--use-ticket-system` - If set, the order of pixels in the output file is deterministic across runs (though not necessarily the same as the input order). It incurs a performance penalty.

`--top-k` - Number of top factors to include in the output. Default: 3.

`--output-coord-digits` - Number of decimal digits to output for coordinates in text mode (used when input coordinates are float or `--output-original` is not set). Default: 2.

`--output-prob-digits` - Number of decimal digits to output for probabilities. Default: 4.

## 3D Modes

3D mode is explicit.

- Use `--icol-z --thin-3D` for thin 3D.
- Use `--icol-z --standard-3D` for standard 3D.
- `--icol-z` without one of those flags is rejected.

### Standard 3D

Standard 3D uses a BCC anchor lattice directly in the input coordinate system.
This means x, y, and z are expected to already use compatible units.

- `--anchor-dist` is the preferred way to define the BCC anchor spacing.
- Each pixel is connected to anchors from a fixed local BCC neighborhood:
  the focal BCC cell plus its 14 face-adjacent neighbors.
- `--radius` does not change that fixed neighborhood, but it does control the
  anchor weight decay scale.
- If `--radius` is omitted, it defaults to `1.2 * (2 * anchor-dist / sqrt(3))`.
- Anchor weights are computed from the actual pixel-to-anchor distance and the
  configured support radius.

At the CLI level, standard 3D does not require `--hex-size`,
`--hex-grid-dist`, or `--n-moves`. The required geometry control is
`--anchor-dist`.

If `--zmin` and `--zmax` are provided in standard 3D, they are ignored unless
`--ignore-outside-zrange` is also set.

Example:

```bash
punkst pixel-decode --model ${path}/bcc.model.tsv \
--in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
--temp-dir ${tmpdir} --out-pref ${path}/pixel_3d --output-binary \
--icol-x 0 --icol-y 1 --icol-z 2 --icol-feature 3 --icol-val 4 \
--standard-3D --anchor-dist 12 \
--pixel-res 0.5 --pixel-res-z 0.5 \
--threads ${threads} --seed 1
```

### Thin 3D

Thin 3D keeps the overlapping shifted hex-anchor construction in x-y and adds a
finite set of z levels across `[zmin, zmax]`.

- `--zmin` and `--zmax` are required.
- One of `--thin-3d-z-levels` or `--thin-3d-n-z-levels` must be provided.
- If both are provided, `--thin-3d-z-levels` takes precedence and
  `--thin-3d-n-z-levels` is ignored with a warning.
- When `--thin-3d-n-z-levels` is used, the z levels are placed evenly between
  `zmin` and `zmax`.
- `--radius` is still used for the pixel-to-anchor graph.
- `--n-init-anchor-per-pix` controls how many nearby z levels each point
  contributes to during anchor initialization.
- The x-y anchor coordinates are still generated from the shifted hex grids.
- The z coordinate of each anchor is assigned by a modular coloring of the
  underlying fine axial lattice, so anchors from different z levels are mixed
  across x-y while staying roughly balanced across levels.

Example:

```bash
punkst pixel-decode --model ${path}/thin3d.model.tsv \
--in-tsv ${path}/transcripts.tiled.tsv --in-index ${path}/transcripts.tiled.index \
--temp-dir ${tmpdir} --out-pref ${path}/pixel_thin3d --output-binary \
--icol-x 0 --icol-y 1 --icol-z 2 --icol-feature 3 --icol-val 4 \
--thin-3D --hex-grid-dist 18 --n-moves 3 \
--thin-3d-z-levels 0 1.5 3 4.5 6 7.5 9 \
--zmin 0 --zmax 20 --n-init-anchor-per-pix 4 \
--pixel-res 0.5 --pixel-res-z 1 \
--threads ${threads} --seed 1
```

### Current 3D Scope

Current 3D support should be understood as a 2D tiled pipeline with 3D-aware
anchor geometry.

- tile ownership and boundary handling are still defined only in x-y
- standard 3D does not support external fixed anchors
- standard 3D requires positive `--anchor-dist`; `--radius` affects weighting
  and padding but not the fixed BCC neighborhood
- thin 3D and standard 3D both use `--half-life-dist` for distance-based
  anchor weighting

## Process multiple samples

If you want to project the same model onto multiple datasets/samples, you can use `--sample-list` to pass a tsv file containing all samples' information. (In this case, `--in-tsv` and `--in-index` are ignored.)
Note: all other parameters are shared across samples, so the input files should have the same structure.

If your multi-sample data are generated by `punkst multisample-prepare`, it has created a file named `*.persample_file_list.tsv`. You can just pass this file to `--sample-list` and optionally use `--out-pref` to specify an identifier (e.g., the model information) to add to each output file name.

If you created the input file for each sample manually, you can create a tsv file with at least three columns:
sample_id, path to the transcript file created by `pts2tiles`, path to the index file created by `pts2tiles`.

Optional fourth column: output prefix.

Optional fifth column: anchor file path.

If there are headers, all header lines should start with "#".

```bash
punkst pixel-decode --model model.tsv --sample-list multi.persample_file_list.tsv \
--temp-dir /tmp --out-pref results --output-binary \
--icol-x 0 --icol-y 1 --icol-feature 2 --icol-val 3 \
--hex-grid-dist 12 --n-moves 2 --threads 8
```

## Output Files

Files are written with the prefix specified by `--out-pref` (or inferred from `--out`).

### Main output

`<prefix>.bin` - Main pixel-level output in binary format when `--output-binary` is set.

`<prefix>.tsv` - Main pixel-level output in text format when `--output-binary` is not set.

`<prefix>.index` - Binary index for the main output. This is written for both text and binary modes. In 2D binary mode, the index describes regular square tiles directly.

`<prefix>.anchors.tsv` - Anchor-level top-factor assignments when `--output-anchors` is set.

### Summary statistics:

`<prefix>.pseudobulk.tsv` - Gene-by-factor pseudobulk matrix.

`<prefix>.confusion.tsv` - Factor-by-factor confusion matrix derived from per-pixel assignments.

`<prefix>.denoised_pseudobulk.tsv` - Denoised version of the pseudobulk matrix.

<!-- `nmf` mode does not currently write these three summary files. -->
