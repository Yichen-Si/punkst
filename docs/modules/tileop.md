# tile-op

**`tile-op` provides utilities to view and manipulate the tiled data files** created by `punkst pixel-decode` or `punkst pts2tiles`.

This module is under active development and any suggestions or requests will be most welcome.

Caution:

Some operations related to smoothing, spatial profiling, and factor masks treat the inference output as a raster multi-channel image where the channel intensities are the factor probabilities. This works well when `pixel-decode` is run with a moderate `--pixel-res`, like `0.5` or `1` for submicron resolution data, or `2` for Visium HD data. If the input was generated at a much finer grid, you can now request a coarser raster grid for these operations with `--raster-pixel-res`.

## Available Operations

- Basic inspection, conversion, and region query
    - [Print Index](#print-index)
    - [Convert TSV](#convert-to-tsv)
    - [Fix Fragmented Tiles](#fix-fragmented-tiles)
    - [Region Query](#region-query)

- Joining, annotating, and aggregation
    - [Merge Multiple Inference Results](#merge-multiple-inference-results)
    - [Annotate Points with Inference Results](#annotate-points-with-inference-results)
    - [Aggregate Results by Cell](#aggregate-results-by-cell)

- Factor-distribution summaries
    - [Compute Joint Probability Distributions](#compute-joint-probability-distributions)
    - [Compute Confusion Matrix](#compute-confusion-matrix)

- Spatial profiling and factor masks
    - [Denoise Top Labels](#denoise-top-labels)
    - [Compute Basic Spatial Metrics](#compute-basic-spatial-metrics)
    - [Shell and Surface Profiles](#shell-and-surface-profiles)
    - [Profile the Area Covered by One Focal Factor](#profile-the-area-covered-by-one-focal-factor)
    - [Soft Factor Mask](#soft-factor-mask)
    - [Soft Mask Composition](#soft-mask-composition)
    - [Hard Factor Mask](#hard-factor-mask)

(Each operation is intended to be used independently, though some operations can be combined to a pipeline, e.g. denoise the factor predictions then profile surface distance; merging multiple inference files before annotating all onto one transcript file)

## Usage

### Main input & output
The main input are the tiled pixel level files created by `punkst pixel-decode` (either in the custom binary format or in plain TSV format) or `punkst pts2tiles`.

You can specify the pair of data and index files using `--in-data` and `--in-index`, or specify the prefix using `--in`.
When using `--in`, without `--binary`, the tool assumes the data file is `<in>.tsv` and the index file is `<in>.index`, and with `--binary` it assumes the data file is `<in>.bin` and the index file is `<in>.index`.

Use `--out` to specify the output prefix. In some operations use `--binary-out` to specify that the output is to be written in binary format.

### Single-molecule Pixel Inputs

If the input was created by `punkst pixel-decode --single-molecule`, each binary record carries an additional feature index in addition to coordinates and factor probabilities.

Current `tile-op` support for these inputs is:

- supported:
  - `--dump-tsv`
  - `--merge-emb`
  - `--annotate-pts`
  - `--annotate-cell`
  - `--prob-dot`
  - `--confusion`
- raster image processing style operations are not supported:
  - `--smooth-top-labels`
  - `--spatial-metrics`
  - `--shell-surface`
  - `--profile-one-factor-mask`
  - `--soft-factor-mask`
  - `--soft-mask-composition`
  - `--hard-factor-mask`

Additional flags used with supported single-molecule workflows:

- `--features`: dictionary file whose first column maps feature indices to feature names
- `--icol-feature`: 0-based feature-name column in the query TSV for `--annotate-pts` and `--annotate-cell`

### Parallel Execution

Most `tile-op` subcommands accept `--threads N`. The current CLI default is `--threads 1`, which keeps all operations single-threaded unless you explicitly request more threads.

Current commands with meaningful parallel tile-local execution include:

- `--merge-emb`
- `--annotate-pts`
- `--prob-dot`
- `--confusion`
- `--extract-region`
- `--extract-region-geojson`
- `--smooth-top-labels`
- `--spatial-metrics`
- `--shell-surface`
- `--profile-one-factor-mask`
- `--soft-factor-mask`
- `--soft-mask-composition`
- `--hard-factor-mask`

Notes:

- Some commands are only partially parallelized: tile-local work is parallel, but later reduction, polygon assembly, or final writeback is still serial.
- `--merge-emb` and `--annotate-pts` use a dedicated writer thread in parallel mode. Their output tiles may appear in an arbitrary order, but indexed lookup is valid.
- `--prob-dot` without `--merge-emb` has a serial fallback for bounded query mode and non-seekable text input such as stdin or gzipped streaming text.

### Raster Resolution Override

Raster-style commands can optionally run on a coarser 2D grid than the source data by passing:

```bash
--raster-pixel-res <value>
```

Current supported commands:

- `--smooth-top-labels`
- `--spatial-metrics`
- `--shell-surface`
- `--profile-one-factor-mask`
- `--soft-factor-mask`
- `--hard-factor-mask`

Rules:

- the requested value is in the original coordinate units, the same units stored in the input header
- it must be strictly larger than and an integer times of the input `pixelResolution`
- all raster-like sizes and distances for the command remain in pixel units, but now refer to the overridden raster grid

### Print Index

To inspect the index of a tiled file (it prints one tile per line after the header, so could be quite long for large data):

```bash
punkst tile-op --print-index --in path/prefix [--binary]
```

### Convert to TSV

To dump a binary tiled file to a plain TSV file:

```bash
punkst tile-op --dump-tsv --in path/prefix --binary --out path/prefix.dump
```

The output include `path/prefix.dump.tsv` and `path/prefix.dump.index`.

For single-molecule binary input, `--features` is required and the dumped TSV includes one extra `feature` column with feature names:

```bash
punkst tile-op --dump-tsv --in path/prefix --binary \
  --features path/features.tsv \
  --out path/prefix.dump
```

### Fix fragmented Tiles

The output of `punkst pixel-decode` is organized into non-overlapping rectangular tiles that jointly cover the entire space, but the tiles do not fit into a regular grid.

If we would need to merge multiple sets of inference results or want to join the inference results with point level data, currently we have to reorganize the data to a regular grid first. (The tile size shoud be already stored in the input's index file (`path/prefix.index`), currently we don't support generic reorganization)

Note that this is **not** required for visualization `draw-pixel-factors`.

```bash
punkst tile-op --reorganize --in path/prefix [--binary] --out path/reorg_prefix
```

### Region Query

You can extract a spatial subset of a tiled file and write it out as another indexed tiled file.

Output:

- `path/prefix.region.tsv` or `path/prefix.region.bin`
- `path/prefix.region.index`

The output remains in regular tiled format even when the query region may partially overlap some tiles, and contains all and only tiles with at least one retained record.

#### Rectangle query

To extract all records inside one axis-aligned rectangle:

```bash
punkst tile-op --extract-region --in path/prefix [--binary] \
  --xmin 1000 --xmax 2000 --ymin 500 --ymax 1500 \
  --out path/prefix.region
```

This keeps all records whose `(x, y)` coordinates fall inside the half-open rectangle `[xmin, xmax) x [ymin, ymax)`.

#### GeoJSON region query

To extract all records inside the union of multiple polygons:

```bash
punkst tile-op --extract-region-geojson path/region.geojson \
  --in path/prefix [--binary] \
  --out path/prefix.region
```

Optional:

- `--extract-region-scale` controls the integer snapping scale used internally for polygon processing. Default is `10`, which corresponds to `0.1` units.

**Requirements**

Tiled input data: GeoJSON region query currently only supports tiled inputs in **regular square tile mode** (see [Fix Fragmented Tiles](#fix-fragmented-tiles) below). The input can be either TSV or binary, but text input must be seekable, so stdin and gzipped streaming text input are not supported for this operation.

GeoJSON / JSON file: see [GeoJSON Region Input](../input/geojson-region.md) for requirements and polygon validity handling.

### Merge Multiple Inference Results

You can merge multiple inference files (e.g., from fitting different models) concerning the same spatial dataset into a single file. The main input (`--in`) defines the output lattice. Auxiliary inputs from `--merge-emb` are matched onto that lattice when they have the same tile size, the same or coarser resolution, and compatible dimensionality.

```bash
punkst tile-op --in path/result1 [--binary] \
  --merge-emb path/result2.tsv path/result3.bin --k2keep 3 1 2 \
  --out path/merged_result --binary-out --threads 1
```

`--merge-emb` - One or more other inference files (created by `pixel-decode`) to merge with the main input file. They can be in either TSV or binary format, but have to have proper index files stored ad `<prefix>.index`.

`--k2keep` - (Optional) A list of integers specifying how many top factors to keep from each source file (including the main input). If not provided, all top in the input files are kept.

`--binary-out` - (Optional) Save the merged output in binary format instead of TSV.

`--merge-keep-all-main` - (Optional) Keep all records from the main input and fill any unmatched auxiliary source slots with `(-1, 0)`. By default, merge keeps only records that match in every source.

`--threads` - (Optional) Number of worker threads for tile-local merge processing.

In the above example, we keep top 3 factors from file `result1.bin` (or `.tsv`), top 1 from `result2.tsv`, and top 2 from `result3.bin`. If the specified number exceeds the number of factors available in the corresponding file, all factors in the file are kept.

For single-molecule inputs, the main input must also be single-molecule. Matching then works as follows:

- if all inputs are single-molecule, records are matched by coordinate and feature index
- if the main input is single-molecule and some later auxiliary inputs are standard pixel outputs, those later inputs are treated as lower-resolution feature-agnostic records and can fan out to multiple feature-bearing main records at the same coordinate
- the merged output keeps the feature index

If the merged output is TSV rather than binary, you may also pass `--features` so the feature-bearing column is written as feature names instead of numeric feature indices.

### Annotate Points with Inference Results

You can annotate a transcript file with the inference results. The query file is required to be generated by `punkst pts2tiles` with the same tile structure as the result file (since you normally run `pixel-decode` with the output from `pts2tiles` as the input), but you can apply `pts2tiles` to any tsv file that contains X, Y coordinates as two of its columns.

```bash
punkst tile-op --in path/prefix [--binary] \
  --annotate-pts path/transcripts --icol-x 0 --icol-y 1 \
  --out path/merged --threads 1
```

`--annotate-pts` - Prefix of the points file (the tool expects `<prefix>.tsv` and `<prefix>.index`) to be annotated.

`--icol-x` - 0-based column index for X coordinate in the points file.

`--icol-y` - 0-based column index for Y coordinate in the points file.

For single-molecule inputs, also provide:

- `--features` - feature dictionary used to map feature names to the feature indices stored in the tiled inference file
- `--icol-feature` - 0-based column index of the feature-name column in the query TSV

Matching is then done by `(x, y, feature)` in 2D or `(x, y, z, feature)` in 3D.

`--threads` - (Optional) Number of worker threads for tile-local annotation.

### Compute Joint Probability Distributions

You can compute the correlations or co-occurrences between factors, either from a single model or between inference results from different models applied to the same dataset. This is approximated by the sum of products of posterior probabilities across all pixels, although for each pixel only the top-K factors are considered (those stored in the inference result file). To compute co-occurrence between factors in a single model at different spatial resolutions, see the confusion matrix operation below.

Note: the pixel level factor probabilities are not to be interpreted as full Bayesian posterior probabilities as they are from approximated computation with mean-field variational inference.

Likely use cases: comparing factor sets; comparing factors with cell types.

#### Single Input

For a single inference result file:

```bash
punkst tile-op --prob-dot --in path/result [--binary] --out path/out_prefix --threads 1
```
Output:

- `path/out_prefix.marginal.tsv`: Marginal sums of probabilities (mass) for each factor. (This should be roughly the same as the auxilliary pseudobulk matrix from `pixel-decode`).

- `path/out_prefix.joint.tsv`: Sum of products for each pair of factors.

`--threads` - (Optional) Number of worker threads for the indexed / seekable-input path.

If the file contains multiple sets of results (e.g. a merged file), the output is the same as the multi-input case below, where it stores marginal and within-model joint output for each source separately, and produces cross-source products (e.g., `path/out_prefix.0v1.cross.tsv`).

#### Merging and Computing on the Fly

You can also compute these statistics while merging multiple inference result files on the fly, without writing the merged file to disk.

```bash
punkst tile-op --prob-dot --in path/result1 [--binary] \
  --merge-emb path/result2.tsv path/result3.bin \
  --out path/out_prefix --threads 1
```

This supports `--k2keep` to reduce the number of top-K factors used in each source before computing the products.

`--threads` - (Optional) Number of worker threads for shared-tile accumulation across input sources.

For this multi-input mode, either all inputs must be standard pixel outputs or all inputs must be single-molecule outputs. In the single-molecule case, records are matched by coordinate plus feature index.

Output:

- `path/out_prefix.0.marginal.tsv`, `path/out_prefix.1.marginal.tsv`, ... (one per input source)

- `path/out_prefix.0.joint.tsv`, ... (internal dot products for each source)

- `path/out_prefix.0v1.cross.tsv`, `path/out_prefix.0v2.cross.tsv`, ... (cross-source dot products with `log10pval` from a naive chi-squared 2x2 enrichment test)

### Compute Confusion Matrix

This operation computes a confusion matrix of factors at a given spatial resolution. It divides the space into squares of a specified size, then builds a matrix of co-occurrences among factors.

```bash
punkst tile-op --confusion 10 --in path/result [--binary] --out path/out_prefix --threads 1
```

`--confusion` - The resolution (side length of square bins in microns) for computing the confusion matrix.

`--threads` - (Optional) Number of worker threads for indexed tile-level confusion accumulation.

For single-molecule input, the feature index is ignored for the confusion definition itself, but every feature-bearing record still contributes to the accumulated matrix.

Output:

- `path/out_prefix.confusion.tsv`: A matrix of co-occurrence counts between factors.

### Aggregate Results by Cell

This operation aggregates pixel-level inference results at cell and subcellular compartment level, based on the (tailed) transcript file that contains cell/compartment annotations per transcript/pixel.
If your data is from CosMx, Xenium, or Visium MERSCOPE, you should have already run `punkst pts2tiles` on the raw transcript file which contains cell ID and possibly a column indicating if the transcript is nuclear or cytoplasmic. Then the tailed file should contain the necessary information.

```bash
punkst tile-op --annotate-cell --in path/result [--binary] \
  --annotate-pts path/transcripts_with_cells \
  --icol-x 0 --icol-y 1 --icol-c 5 --icol-s 6 \
  --out path/cellular_results
```

This command will summarize the factors for each cell ID found in `path/transcripts_with_cells.tsv`.

`--annotate-cell` - Flag to enable aggregation by cell.

`--annotate-pts` - Prefix of the points file (e.g. transcripts) containing cell annotations.

`--icol-x`, `--icol-y` - 0-based column indices for X and Y coordinates.

`--icol-z` - (Optional) 0-based column index for Z coordinate.

For single-molecule inputs, also provide:

- `--features` - feature dictionary used to map feature names to feature indices
- `--icol-feature` - 0-based feature-name column in the annotated query TSV

Matching is then done by coordinate plus feature before factor probabilities are aggregated per cell.

`--icol-c` - 0-based column index for the cell ID.

`--icol-s` - (Optional) 0-based column index for subcellular component annotations. If provided, results will be aggregated per-cell and per-component.

`--k-out` - (Optional) Number of top factors to include in the output for each cell/component. If not provided, the same number of in the input file is used.

`--max-cell-diameter` - (Optional) The maximum expected diameter of a cell in microns. Used for avoiding boundary effects as we process by tiles. Default is 50.

Output:

A TSV file `path/cellular_results.tsv` containing aggregated factor probabilities for each cell (and component, if specified).

A TSV file `path/cellular_results.pseudobulk.tsv` containing the sum of factor probabilities across each subcellular component. Useful for comparing global factor abundance between components.

### Denoise Top Labels

This is a heuristic denoising operation on the top-predicted factor labels for each pixel. It replace pixels where the predicted factor differs from most of its neighbors with the majority vote among its neighbors.
It is meant for the case where you projected categorical cell types at high resolution data where you do not expect to see much mixing of cell types at single pixel level.
The output is a new tiled data file where for each pixel, only the smoothed top factor is kept. (The output can be used as input for `tile-op`, so you can dump it to a tsv file or do other operations)

```bash
punkst tile-op --smooth-top-labels 2 --in path/result [--binary] --out path/smoothed_result
```

`--smooth-top-labels` - The number of rounds to perform the denoising operation. A value greater than 0 enables the operation. One or two rounds is usually sufficient.

Optional:

`fill-empty-islands` - fill isolated empty pixels if they are surrounded by consistent neighbors. Default is to leave empty pixels unchanged. This may be helpful if you would like to get statistics like area and perimeter/edge per cell type later using `tile-op --spatial-metrics`

`--raster-pixel-res` - (Optional) run smoothing on a coarser raster grid. The output header records the requested pixel resolution.

### Compute Basic Spatial Metrics

This is more interpretable for cell type/cluster projection (so the labels are categorical). It is recommended to denoise and fill in scattered empty pixels first with `tile-op --smooth-top-labels r --fill-empty-islands` (see above).

```bash
punkst tile-op --spatial-metrics --in path/result [--binary] --out path/prefix
```

The output includes two files:

- `path/prefix.stats.single.tsv` for per-channel (factor or cell type) metrics. Columns are:
  - channel index (`#k`)
  - total number of pixels (`area`)
  - total length of all pixel-to-pixel boundaries shared with other channels, including the explicit background channel `K` (`perim`)

The single-channel table now includes one extra row with `#k = K`, representing empty/background area.

- `path/prefix.stats.pairwise.tsv` for pairwise metrics between channels. Let the areas and non-boundary perimeters a pair of channels be $A_k, P_k, A_l, P_l$, the columns are
  - channel indices for the pair (`#k`, `l`)
  - length of shared boundary $L_{kl}$ (`contact`)
  - $L_{kl} / (P_k + P_l - L_{kl})$ (`frac0`)
  - $L_{kl} / P_k$ (`frac1`)
  - $L_{kl} / P_l$ (`frac2`)
  - $L_{kl} / (A_k + A_l)$ (`density`)

### Shell and Surface Profiles

Profile the factor composition in the immediate neighborhood of a factor, and pairwise spatial proximity between factors.

Here we only use the top predicted factor for each pixel, so the masks for factors are mutually exclusive.

Shell composition: consider each focal factor as defining a binary mask, we first find the contour of each patch (connected component) of the foreground of the mask, a shell is defined as the set of pixels within a certain distance from the contour. For each of the specified shell radii, we report the composition of other factors within the shell.

Surface distance: for each pair of factors, we compute a histogram of the distance from pixels of one factor to the nearest pixel of the other factor, and vice versa. This is a directional measure of spatial proximity between factors. It is approximated for efficiency and robustness by first extracting boundaries of the factor masks, then computing the distance from each boundary pixel to the nearest pixel on the other factor's boundary. The output is a histogram of these distances for each pair of factors with bin size `1` and up to the specified maximum distance.

**CAUTION**: all length and area parameters are in pixel units because this operation views the data as a rasterized image. By default those pixels are the input pixels. If you pass `--raster-pixel-res`, the same parameters are interpreted on that coarser raster grid instead.

```bash
punkst tile-op --shell-surface --in path/result [--binary] \
  --shell-radii 5 10 20 --surface-dmax 25 \
  --cc-min-size 25 --spatial-min-pix-per-tile-label 20 \
  --out path/out_prefix
```

`--shell-surface` - run shell occupancy and surface distance profiling.

`--shell-radii` - one or more shell radii (in pixels, NOT microns) for occupancy reporting.

`--surface-dmax` - maximum distance bin (in pixels) for the surface-distance histogram.

`--cc-min-size` - minimum connected-component size (number of pixels) used for boundary seed filtering.

`--spatial-min-pix-per-tile-label` - require at least this many pixels of a label within a tile before that tile contributes to this label's boundary construction.

`--raster-pixel-res` - (Optional) run shell and surface profiling on a coarser raster grid.

Output:

- `path/out_prefix.shell.tsv`: shell occupancy summary with columns
  - focal label (`#K_focal`)
  - other label (`K2`)
  - radius (`r`)
  - number of `K2` pixels within distance `r` from boundary of focal label (`n_within`)
  - total number of `K2` pixels (`n_K2_total`)

- `path/out_prefix.surface.tsv`: directional surface-distance histogram with columns
  - source label (`#from_K1`)
  - target label (`to_K2`)
  - distance bin (`d`)
  - number of `K1` boundary pixels that find the neraest boundary of `K2` at distance `d` (`count`)

### Profile the area covered by one focal factor

Build a raster mask for one focal factor using local neighborhood probability mass, optionally remove isolated small spots/patches, then report the factor composition inside the mask. Optionally, it then creates a soft mask for each of the factors with a high total probability mass inside the focal mask and calculate pairwise overlaps among the focal and selected factor masks.
(It currently does not output the boundaries. See `--soft-factor-mask` below for that)

```bash
punkst tile-op --profile-one-factor-mask --in path/result [--binary] \
  --focal-k 7 --mask-radius 2 --mask-threshold 0.35 \
  --mask-min-frac 0.05 --mask-min-component-area 20 \
  --out path/out_prefix
```

Main parameters:

- `--focal-k` - focal factor index.
- `--mask-radius` - size `r` (in pixel units) defining the `(2r+1) x (2r+1)` neighborhood.
- `--mask-threshold` - threshold on the focal factor neighborhood score.
- `--mask-min-frac` - keep a secondary factor if its mass inside the focal mask exceeds this fraction of the total focal-mask mass.
- `--mask-min-pixel-prob` - optional per-pixel cutoff used only when constructing masks from factor probabilities.
- `--mask-morphology` - optional post-threshold morphology sequence. Each value is an odd kernel size with sign indicating the operation: positive for dilation, negative for erosion. For example, `--mask-morphology 5 -3`.
- `--mask-min-component-area` - optional 4-connected component size cutoff applied independently within each tile after thresholding.
- `--raster-pixel-res` - (Optional) run the mask construction and overlap profiling on a coarser raster grid.

Output:

- `path/out_prefix.factor_hist.tsv`: factor histogram for both the focal mask and the full processed region with columns
  - factor index (`k`)
  - total mass inside the focal mask (`mass_in_mask`)
  - fraction of the total focal-mask mass (`frac_in_mask`)
  - total mass in the full processed region (`mass_global`)
  - fraction of the total global mass (`frac_global`)

- `path/out_prefix.pairwise.tsv`: pairwise overlap summary for the selected factor set `{focal_k} U {significant secondary factors}` with columns
  - factor indices (`k1`, `k2`)
  - mask areas (`area1_pix`, `area2_pix`)
  - area intersection (`area_ovlp_pix`)
  - directional area overlap fractions (`area_ovlp_f1`, `area_ovlp_f2`)
  - area Jaccard (`area_jaccard`)
  - factor-specific mass in the intersection (`mass1_in_ovlp`, `mass2_in_ovlp`)
  - directional mass overlap fractions relative to each factor's total global mass (`mass_ovlp_f1`, `mass_ovlp_f2`)

### Soft factor mask

Build a soft binary mask for every factor, optionally remove small connected components, and write global summaries. By default this also polygonizes the kept mask and exports one GeoJSON `MultiPolygon` feature per factor; use `--skip-boundaries` to disable geometry export.

```bash
punkst tile-op --soft-factor-mask --in path/result [--binary] \
  --mask-radius 2 --mask-threshold 0.35 \
  --mask-min-pixel-prob 0.01 --mask-min-tile-mass 2 \
  --mask-min-component-area 20 --mask-min-hole-area 4 \
  --mask-simplify 2 --out path/out_prefix
```

Main parameters:

- `--mask-radius` - size `r` (in pixel units) defining the `(2r+1) x (2r+1)` neighborhood.
- `--mask-threshold` - threshold on the factor probabilities averaged over the observed pixels in the neighborhood. Pixels with no observation do not contribute to the denominator. For an empty center pixel, if at most 2 of its 4 direct neighbors are observed, it is excluded immediately; otherwise it is evaluated by the same mass-based rule as any other pixel. The window must also contain at least total factor mass `1.0`.
- `--mask-min-pixel-prob` - ignore sparse factor entries below this per-pixel probability before mask construction.
- `--mask-morphology` - optional post-threshold morphology sequence. Each value is an odd kernel size with sign indicating the operation: positive for dilation, negative for erosion. For example, `--mask-morphology 5 -3`.
- `--mask-min-tile-mass` - skip a factor in a tile if its retained sparse mass in that tile is below this threshold.
- `--mask-min-component-area` - legacy-named cutoff applied independently within each tile after thresholding; in `--soft-factor-mask` it is compared against the total raw factor mass in each 4-connected component, not the pixel area.
- `--mask-min-hole-area` - drop holes smaller than this area from the polygon output.
- `--mask-simplify` - optional [Clipper2 SimplifyPaths](https://www.angusj.com/clipper2/Docs/Units/Clipper/Functions/SimplifyPaths.htm) tolerance; `0` keeps the exact raster-derived boundary after collinear trimming (which may have staircase-like boundaries due to rasterization). The unit is in pixel.
- `--skip-boundaries` - skip GeoJSON generation and write only the summary tables.
- `--template-geojson` - optional template GeoJSON file. When provided, `tile-op` still writes the generic `FeatureCollection` GeoJSON and also writes one extra GeoJSON file per factor. The template's top-level metadata is preserved, `title` is set to the factor index, and the GeoJSON payload is replaced with a single factor-specific feature/geometry in a GeoJSON-valid way.
- `--template-out-prefix` - optional output prefix for the per-factor template-derived GeoJSON files. Defaults to `--out`.
- `--raster-pixel-res` - (Optional) run mask construction and polygonization on a coarser raster grid.

Output:

- `path/out_prefix.factor_summary.tsv`: per-factor summary with columns
  - factor index (`k`)
  - number of tiles where the factor passed `--mask-min-tile-mass` (`n_tiles`)
  - total kept soft-mask area in pixels (`mask_area_pix`)
  - number of final connected components after seam merge (`n_components`)

- `path/out_prefix.component_hist.tsv`: final component-size histogram with columns
  - factor index (`k`)
  - component size (`size`)
  - number of components with that size (`n_components`)

- `path/out_prefix.geojson`: optional `FeatureCollection` containing one `MultiPolygon` feature per factor with properties
  - factor index (`Factor`)
  - number of contributing tiles (`n_tiles`)
  - total kept mask area in pixels (`mask_area_pix`)
  - number of final connected components (`n_components`)

- `path/out_prefix.k<factor>.geojson`: optional per-factor GeoJSON files written only when `--template-geojson` is provided. If `--template-out-prefix` is set, that prefix is used instead of `out_prefix`. Each file preserves the template's top-level metadata, sets `title` to the factor index, and replaces the template's GeoJSON payload with the corresponding per-factor boundary.

### Soft mask composition

Read the joined GeoJSON produced by `--soft-factor-mask`, treat each feature as one focal-factor mask, and compute the factor composition inside each mask as well as globally over the full processed input. Masks may overlap, so one pixel can contribute to multiple focal-mask histograms.

```bash
punkst tile-op --soft-mask-composition path/out_prefix.geojson \
  --soft-mask-composition-focal 3 7 \
  --in path/result [--binary] \
  --out path/out_prefix
```

Main parameters:

- `--soft-mask-composition` - path to the joined GeoJSON written by `--soft-factor-mask`.
- `--soft-mask-composition-focal` - optional subset of focal factor IDs to profile from the GeoJSON. If not provided, all valid factor masks will be profiled. Duplicate IDs are ignored with a warning. The global histogram is always included in the output.

Output:

- `path/out_prefix.mask_composition.tsv`: mask and global factor histograms with columns
  - focal factor index (`k_focal`)
  - factor index (`k`)
  - total probability mass (`mass`)
  - fraction of the total mass for that focal mask (`frac`)

For the global histogram block, `k_focal` is written as `K`, the total number of factors in the input.

### Hard factor mask

Build per-label hard masks from the top predicted factor at each raster pixel, merge connected components across tile boundaries, and write global summaries. By default this also writes one GeoJSON `MultiPolygon` feature per factor; use `--skip-boundaries` to disable boundary extraction and write only the summaries.

```bash
punkst tile-op --hard-factor-mask --in path/result [--binary] \
  --cc-min-size 25 --out path/out_prefix
```

Main parameters:

- `--cc-min-size` - minimum final connected-component size retained in the summaries and GeoJSON.
- `--skip-boundaries` - skip GeoJSON generation and write only the summary tables.
- `--template-geojson` - optional template GeoJSON file. When provided, `tile-op` still writes the generic `FeatureCollection` GeoJSON and also writes one extra GeoJSON file per factor. The template's top-level metadata is preserved, `title` is set to the factor index, and the GeoJSON payload is replaced with a single factor-specific feature/geometry in a GeoJSON-valid way.
- `--template-out-prefix` - optional output prefix for the per-factor template-derived GeoJSON files. Defaults to `--out`.
- `--raster-pixel-res` - (Optional) build the hard-label raster and connected components on a coarser grid.

Output:

- `path/out_prefix.factor_summary.tsv`: per-factor summary with columns
  - factor index (`k`)
  - number of tiles containing the factor (`n_tiles`)
  - total retained mask area in pixels (`mask_area_pix`)
  - number of retained final connected components (`n_components`)

- `path/out_prefix.component_hist.tsv`: retained component-size histogram with columns
  - factor index (`k`)
  - component size (`size`)
  - number of components with that size (`n_components`)

- `path/out_prefix.geojson`: optional `FeatureCollection` containing one `MultiPolygon` feature per factor with properties
  - factor index (`Factor`)
  - number of contributing tiles (`n_tiles`)
  - total retained mask area in pixels (`mask_area_pix`)
  - number of retained final connected components (`n_components`)

- `path/out_prefix.k<factor>.geojson`: optional per-factor GeoJSON files written only when `--template-geojson` is provided. If `--template-out-prefix` is set, that prefix is used instead of `out_prefix`. Each file preserves the template's top-level metadata, sets `title` to the factor index, and replaces the template's GeoJSON payload with the corresponding per-factor boundary.
