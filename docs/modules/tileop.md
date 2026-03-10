# tile-op

**`tile-op` provides utilities to view and manipulate the tiled data files** created by `punkst pixel-decode` or `punkst pts2tiles`.

## Available Operations

- inspecting the index

- converting binary tiled files to TSV

- extracting records inside a rectangular or multi-polygon query region

- reorganizing fragmented tiles to a regular grid

- merging multiple inference results

- annotating (tiled) point level file with inference results

- compute joint probability distributions of factors

- compute confusion matrix among factors at a given resolution

- denoise and keep only the top predicted factor per pixel

- aggregate pixel level inference results by cell and subcellular compartments based in transcript/pixel level annotations

- compute global connected components per label with centroid and coordinate range

- profile shell occupancy and directional surface distance between labels

- profile the area (softly) covered by a focal factor

- build per-factor masks and export boundaries as GeoJSON

(Except for printing the index, all operations are intended to be used separately)

## Usage

### Main input & output
The main input are the tiled pixel level files created by `punkst pixel-decode`, either in the custom binary format or in plain TSV format.

You can specify the pair of data and index files using `--in-data` and `--in-index`, or specify the prefix using `--in`.
When using `--in`, without `--binary`, the tool assumes the data file is `<in>.tsv` and the index file is `<in>.index`, and with `--binary` it assumes the data file is `<in>.bin` and the index file is `<in>.index`.

Use `--out` to specify the output prefix. In some operations use `--binary-out` to specify that the output is to be written in binary format.

### Basic Inspection and Conversion

To inspect the index of a tiled file:

```bash
punkst tile-op --print-index --in path/prefix [--binary]
```

To dump a binary tiled file to a plain TSV file:

```bash
punkst tile-op --dump-tsv --in path/prefix --binary --out path/prefix.dump
```

The output include `path/prefix.dump.tsv` and `path/prefix.dump.index`.

### Region Query

You can extract a spatial subset of a tiled file and write it out as another indexed tiled file pair.

Output:

- `path/prefix.region.tsv` or `path/prefix.region.bin`
- `path/prefix.region.index`

The output remains in regular tiled format and contains only tiles with at least one retained record.

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

Tiled input data: GeoJSON region query currently only supports tiled inputs in **regular tile mode**:

- the input index must have `mode & 0x8 == 0`, generic rectangular block mode is rejected
- the input can be either TSV or binary, but text input must be seekable, so stdin and gzipped streaming text input are not supported for this operation

GeoJSON / JSON file: see [GeoJSON Region Input](../input/geojson-region.md) for requirements and polygon validity handling.

### Fix fragmented Tiles

The output of `punkst pixel-decode` is organized into non-overlapping rectangular tiles that jointly cover the entire space, but the tiles do not fit into a regular grid.

If we would need to merge multiple sets of inference results or want to join the inference results with point level data, currently we have to reorganize the data to a regular grid first. (The tile size shoud be already stored in the input's index file (`path/prefix.index`), currently we don't support generic reorganization)

Note that this is **not** required for visualization `draw-pixel-factors`.

```bash
punkst tile-op --reorganize --in path/prefix [--binary] --out path/reorg_prefix
```

### Merge Multiple Inference Results

You can merge multiple inference files (e.g., from different models) into a single file. This finds the intersection of tiles and concatenates the results ((factor, probability) pairs) for each pixel.

```bash
punkst tile-op --in path/result1 [--binary] \
  --merge-emb path/result2.tsv path/result3.bin --k2keep 3 1 2 \
  --out path/merged_result --binary-out
```

`--merge-emb` - One or more other inference files (created by `pixel-decode`) to merge with the main input file. They can be in either TSV or binary format, but have to have proper index files stored ad `<prefix>.index`.

`--k2keep` - (Optional) A list of integers specifying how many top factors to keep from each source file (including the main input). If not provided, all factors are kept.

`--binary-out` - (Optional) Save the merged output in binary format instead of TSV.

In the above example, from file `result1.bin` (or `.tsv`) we keep top 3 factors, from `result2.tsv` we keep top 1 factor, and from `result3.bin` we keep top 2 factors. If the specified number exceeds the number of factors available in the corresponding file, all factors in the file are kept.

### Annotate Points with Inference Results

You can annotate a transcript file with the inference results. The query file is required to be generated by `punkst pts2tiles` with the same tile structure as the result file so that the tool can efficiently join it with the inference results, but you can apply `pts2tiles` to any tsv file that contains X, Y coordinates as two of its columns.

```bash
punkst tile-op --in path/prefix [--binary] \
  --annotate-pts path/transcripts --icol-x 0 --icol-y 1 \
  --out path/merged
```

`--annotate-pts` - Prefix of the points file (the tool expects `<prefix>.tsv` and `<prefix>.index`) to be annotated.

`--icol-x` - 0-based column index for X coordinate in the points file.

`--icol-y` - 0-based column index for Y coordinate in the points file.

### Compute Joint Probability Distributions

You can compute the correlations or co-occurrences between factors, either from a single model or between inference results from multiple models applied to the same dataset. This is approximated by the sum of products of posterior probabilities across all pixels, although for each pixel only the top-K factors are considered (those stored in the inference result file).
To compute co-occurrence between factors in a single model at different spatial resolutions, see the confusion matrix operation below.


#### Single Input

For a single inference result file:

```bash
punkst tile-op --prob-dot --in path/result [--binary] --out path/out_prefix
```
Output:

- `path/out_prefix.marginal.tsv`: Marginal sums of posterior probabilities for each factor.

- `path/out_prefix.joint.tsv`: Sum of products for each pair of factors.

If the file contains multiple sets of results (e.g. a merged), the output is the same as the multi-input case below, where it stores marginal and within-model joint output for each source separately, and produces cross-source products (e.g., `path/out_prefix.0v1.cross.tsv`).

#### Merging and Computing on the Fly

You can also compute these statistics while merging multiple inference result files on the fly, without writing the large merged file to disk.

```bash
punkst tile-op --prob-dot --in path/result1 [--binary] \
  --merge-emb path/result2.tsv path/result3.bin \
  --out path/out_prefix
```

This supports `--k2keep` to reduce the number of top-K factors used in each source before computing the products.

Output:

- `path/out_prefix.0.marginal.tsv`, `path/out_prefix.1.marginal.tsv`, ... (one per input source)

- `path/out_prefix.0.joint.tsv`, ... (internal dot products for each source)

- `path/out_prefix.0v1.cross.tsv`, `path/out_prefix.0v2.cross.tsv`, ... (cross-source dot products with `log10pval` from a naive chi-squared 2x2 enrichment test)

### Compute Confusion Matrix

This operation computes a confusion matrix of factors at a given spatial resolution. It divides the space into squares of a specified size, identifies the top factor for each square, and then builds a matrix of co-occurrences.

```bash
punkst tile-op --confusion 10 --in path/result [--binary] --out path/out_prefix
```

`--confusion` - The resolution (side length of square bins in microns) for computing the confusion matrix.

Output:

- `path/out_prefix.confusion.tsv`: A matrix of co-occurrence counts between factors.

### Aggregate Results by Cell

This operation aggregates pixel-level inference results at cell and subcellular compartment level, based on the tailed transcript file that contains cell/compartment annotations per transcript/pixel.
If your data is from CosMx, Xenium, or Visium MERSCOPE, you should have run `punkst pts2tiles` on the raw transcript file which contains cell ID and possibly a column indicating if the transcript is nuclear or cytoplasmic. Then the tailed file already contains the necessary information.

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

### Compute basic spatial metrics

This is more interpretable for cell type/cluster projection (so the labels are categorical). It is recommended to denoise and fill in scattered empty pixels first with `tile-op --smooth-top-labels r --fill-empty-islands` (see above).

```bash
punkst tile-op --smooth-top-labels 2 --in path/result [--binary] --out path/prefix --spatial-metrics
```

The output includes two files:

- `path/prefix.stats.single.tsv` for per-channel (factor or cell type) metrics. Columns are:
  - channel index (`#k`)
  - total number of pixels (`area`)
  - length of all boundaries shared with non-background pixels from other channels (`perim`)
  - length of boundaries shared with background pixels (`perim_bg`)

- `path/prefix.stats.pairwise.tsv` for pairwise metrics between channels. Let the areas and non-boundary perimeters a pair of channels be $A_k, P_k, A_l, P_l$, the columns are
  - channel indices for the pair (`#k`, `l`)
  - length of shared boundary $L_{kl}$ (`contact`)
  - $L_{kl} / (P_k + P_l - L_{kl})$ (`frac0`)
  - $L_{kl} / P_k$ (`frac1`)
  - $L_{kl} / P_l$ (`frac2`)
  - $L_{kl} / (A_k + A_l)$ (`density`)

### Hard factor mask

Build per-label hard masks from the top predicted factor at each raster pixel, merge connected components across tile boundaries, and write global summaries. By default this also writes one GeoJSON `MultiPolygon` feature per factor; use `--skip-boundaries` to disable boundary extraction and write only the summaries.

```bash
punkst tile-op --hard-factor-mask --in path/result [--binary] \
  --cc-min-size 25 --out path/out_prefix
```

Main parameters:

- `--cc-min-size` - minimum final connected-component size retained in the summaries and GeoJSON.
- `--skip-boundaries` - skip GeoJSON generation and write only the summary tables.
- `--template-geojson` - optional template JSON/GeoJSON file. When provided, `tile-op` still writes the generic `FeatureCollection` GeoJSON and also writes one extra JSON file per factor with the template content, replacing only the top-level `title` and `geometry` fields. The title is set to the factor index, and `geometry` is set to the corresponding per-factor GeoJSON `Feature`.
- `--template-out-prefix` - optional output prefix for the per-factor template-derived files. Defaults to `--out`.

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

- `path/out_prefix.k<factor>.json`: optional per-factor JSON files written only when `--template-geojson` is provided. If `--template-out-prefix` is set, that prefix is used instead of `out_prefix`. Each file copies the template and replaces:
  - `title` with the factor index
  - `geometry` with the corresponding per-factor GeoJSON `Feature`

### Shell and Surface Profiles

Profile pairwise spatial proximity between labels using boundary-seeded distance transforms.

```bash
punkst tile-op --shell-surface --in path/result [--binary] \
  --shell-radii 5 10 20 --surface-dmax 25 \
  --cc-min-size 25 --spatial-min-pix-per-tile-label 20 \
  --out path/out_prefix
```

`--shell-surface` - run shell occupancy and surface distance profiling.

`--shell-radii` - one or more shell radii (pixels) for occupancy reporting.

`--surface-dmax` - maximum distance bin for the surface-distance histogram.

`--cc-min-size` - minimum connected-component size used for boundary seed filtering.

`--spatial-min-pix-per-tile-label` - require at least this many pixels of a label within a tile before that tile contributes seeds for the label.

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
  - count (`count`)

### Profile factor masks

Build a raster mask for one focal factor using local neighborhood probability mass, optionally remove small connected components, then report the factor composition inside the focal mask and pairwise overlaps among the selected factor masks.

```bash
punkst tile-op --profile-factor-masks --in path/result [--binary] \
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
- `--mask-min-component-area` - optional 4-connected component size cutoff applied independently within each tile after thresholding.

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

Build a soft binary mask for every factor, remove small connected components, merge seam-crossing components across tiles, and write global summaries. By default this also polygonizes the kept mask and exports one GeoJSON `MultiPolygon` feature per factor; use `--skip-boundaries` to disable geometry export.

```bash
punkst tile-op --soft-factor-mask --in path/result [--binary] \
  --mask-radius 2 --mask-threshold 0.35 \
  --mask-min-pixel-prob 0.01 --mask-min-tile-mass 2 \
  --mask-min-component-area 20 --mask-min-hole-area 4 \
  --mask-simplify 2 --out path/out_prefix
```

Main parameters:

- `--mask-radius` - size `r` (in pixel units) defining the `(2r+1) x (2r+1)` neighborhood.
- `--mask-threshold` - threshold on the factor probabilities averaged over the observed pixels in the neighborhood. Pixels with no observation do not contribute to the denominator. An empty center pixel is still allowed into the mask unless observed coverage in its clipped neighborhood falls below half. The window must also contain at least total factor mass `1.0`.
- `--mask-min-pixel-prob` - ignore sparse factor entries below this per-pixel probability before mask construction.
- `--mask-min-tile-mass` - skip a factor in a tile if its retained sparse mass in that tile is below this threshold.
- `--mask-min-component-area` - legacy-named cutoff applied independently within each tile after thresholding; in `--soft-factor-mask` it is compared against the total raw factor mass in each 4-connected component, not the pixel area.
- `--mask-min-hole-area` - drop holes smaller than this area from the polygon output.
- `--mask-simplify` - optional Clipper2 simplification tolerance; `0` keeps the exact raster-derived boundary after collinear trimming.
- `--skip-boundaries` - skip GeoJSON generation and write only the summary tables.
- `--template-geojson` - optional template JSON/GeoJSON file. When provided, `tile-op` still writes the generic `FeatureCollection` GeoJSON and also writes one extra JSON file per factor with the template content, replacing only the top-level `title` and `geometry` fields. The title is set to the factor index, and `geometry` is set to the corresponding per-factor GeoJSON `Feature`.
- `--template-out-prefix` - optional output prefix for the per-factor template-derived files. Defaults to `--out`.

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

- `path/out_prefix.k<factor>.json`: optional per-factor JSON files written only when `--template-geojson` is provided. If `--template-out-prefix` is set, that prefix is used instead of `out_prefix`. Each file copies the template and replaces:
  - `title` with the factor index
  - `geometry` with the corresponding per-factor GeoJSON `Feature`
