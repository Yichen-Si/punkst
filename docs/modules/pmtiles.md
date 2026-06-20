# pmtiles

**PMTiles utilities in `punkst` currently cover:**

- [Export point-only PMTiles as TSV](#export-pmtiles-as-tsv)
- [Write mono raster PMTiles from tiled counts](#write-mono-raster-pmtiles)
- [Write single-level polygon-only PMTiles](#write-single-zoom-polygon-pmtiles)
- [Write generic polygon PMTiles from arbitrary properties](#write-generic-polygon-pmtiles-from-arbitrary-properties)
- [Build PMTiles pyramids](#build-pmtiles-pyramids)

Point and simple-polygon PMTiles support both [MLT](https://maplibre.org/maplibre-tile-spec/) and MVT.

For writing point/pixel-only PMTiles, see [tile-op](tileop.md).

Acknowledgement: PMTiles support in `punkst` depends on [Clipper2](https://github.com/AngusJohnson/Clipper2) for polygon operations, borrows from [tippecanoe](https://github.com/felt/tippecanoe) (which supports MVT) and [MLT's cpp implementation](https://github.com/maplibre/maplibre-tile-spec/tree/main/cpp).

## Export PMTiles as TSV

`punkst export-pmtiles` reads an MLT- or MVT-backed PMTiles archive. Point archives export back to a plain TSV plus `.index` that `tile-op` can read again. Simple-polygon archives export to a TSV with centroid, vertices, optional feature ID, and properties.

```bash
punkst export-pmtiles \
  --in path/prefix.pmtiles \
  --tile-size 500 \
  --out path/prefix
```

Requirements and behavior:

- `--in` or `--in-data` must point to a point PMTiles archive with **MLT** or **MVT** format
- `--out` specifies the output prefix. The output includes `path/prefix.tsv` and `path/prefix.index`
- `--tile-size` is required and defines the tile size for the output file (in the original units)
- by default, data is read from the archive's max-zoom level
- use `--zoom <z>` to export a specific PMTiles zoom level; the command errors if the archive has no tiles at that zoom
- output columns are `x`, `y`, optional `z`, then the decoded PMTiles schema columns
- missing `K` and `P` values are rendered as `-1` and `0`; other nullable fields are rendered as `NA`


Region filters are supported in this mode:

- `--xmin`, `--xmax`, `--ymin`, `--ymax`
- `--extract-region-geojson`
- `--zmin`, `--zmax`

For simple-polygon archives, add `--polygon`. Polygon export writes only `path/prefix.tsv`; `--tile-size` is not required. `--zoom` is also supported for polygon export.

## Write mono raster PMTiles

`punkst tiles2mono` writes grayscale PNG raster PMTiles from a tiled transcript/count TSV produced by `pts2tiles`. It is used by `deploy-cartoscope` to generate the SGE mono basemap.

```bash
punkst tiles2mono \
  --in path/transcripts.tiled \
  --min-zoom 7 \
  --max-zoom 18 \
  --display-transform linear \
  --out path/sge-mono-dark.pmtiles \
  --threads 4
```

Options:

- `--in` is shorthand for `--in-tsv path/prefix.tsv --in-index path/prefix.index`; if available, `path/prefix.coord_range.tsv` is used for raster bounds
- `--icol-x`, `--icol-y`, and `--icol-count` select the 0-based x, y, and count columns (defaults: `0`, `1`, `3`)
- `--min-zoom` and `--max-zoom` set the raster PMTiles zoom range
- `--adjust-quantile` controls per-zoom density auto-adjustment (default: `0.99`); use `--no-auto-adjust` to write capped raw count intensities
- `--display-transform` controls the final grayscale mapping after auto-adjustment: `linear` (default) or `log1p`
- by default, raw data is parsed only for `--max-zoom`; lower zoom levels are derived by summing child pixels into parent pixels while preserving full count dynamic range until final PNG encoding
- `--max-zoom-from-raw` parses raw data for zooms greater than or equal to the given value, then derives lower zooms from parent layers; set it to `--min-zoom` to parse every zoom from raw data

## Write single-zoom polygon PMTiles

`punkst poly2pmtiles` writes a **polygon-only** MLT or MVT PMTiles archive with **a single zoom level** from a factor-probability table (output from [`punkst topic-model`](lda4hex.md) or `lda-transform`). Use `--format MVT` for MVT output; the default is `MLT`.

It supports two input styles:

- hexagon mode: build polygon geometry from hex centers
- generic polygon mode: link factor-probability rows to polygon geometry by polygon ID

### Shared settings

These options apply to both hexagon mode and generic simple-polygon mode.

#### Shared factor table settings

- `--in-tsv` points to the factor-probability table
- `--out` sets the output PMTiles archive
- `--format` can be `MLT` or `MVT` (default: `MLT`)
- `--pmtiles-zoom` (required) sets the output zoom level
- `--topk-col` and `--topp-col` set column names for the top factor ID and probability (default: `topK` and `topP`)
- compact `K1/P1`, `K2/P2`, ... columns are recognized automatically
- dense numeric factor columns named `0..K-1` are recognized automatically
- `--factor-col-begin` and `--factor-col-end` explicitly select dense factor probability columns by a 0-based inclusive column range; selected columns are mapped to factor IDs `0..K-1` in column order
- `--top-k` controls how many top factors are retained from dense factor columns (default: `3`)
- `--prob-thres` keeps only factor probabilities above the threshold as nullable properties (default: `1e-4`)
- `--coord-scale` scales input coordinates before Web Mercator tiling (default: `1`, no scaling)
- `--layer-name` optionally sets the PMTiles layer name (default: basename of `--out`)
- `--threads` sets the number of encode threads

#### Polygon tiling behavior

These options are similar to [tippecanoe's clipping options](https://github.com/felt/tippecanoe?tab=readme-ov-file#controlling-clipping-to-tile-boundaries):

- `--tile-buffer-px` sets the screen-pixel buffer used by the default clipped/duplicated mode (default: `5`) (matching `tippecanoe -b`)
- `--no-clipping` duplicates each polygon into every touched tile without clipping it to tile boundaries (matching `tippecanoe -pc`)
- `--no-duplication` stores each polygon intact in exactly one tile at the requested zoom instead of clipping and duplicating it across tile boundaries (matching `tippecanoe -pD`)
- `--no-clipping` and `--no-duplication` are mutually exclusive
- `--clip-scale` sets the integer scale used internally for polygon clipping
- `--extent` sets the vector tile extent

### Hexagon mode specific settings

Use this when the input table contains `x` and `y` coordinates as hexagon centers.

```bash
punkst poly2pmtiles \
  --in-tsv path/model.results.tsv.gz \
  --hex-grid-dist 12 \
  --pmtiles-zoom 18 \
  --out path/model.hex.z18.pmtiles
```

Options:

- `--hex-grid-dist` (required) specifies the distance between adjacent grid points on the hexagonal grid
- `--x-col` and `--y-col` set column names for the hex center coordinates (default: `x` and `y`)

### Generic simple-polygon mode with geometry file

Use this when the factor-probability table contains a polygon ID column and polygon geometry is provided in a separate file. For example, if you ran `punkst topic-model` on segmented cells and you want to create a PMTiles archive with cell boundaries as geometry.

```bash
punkst poly2pmtiles \
  --in-tsv path/cells.results.tsv \
  --id-col cell_id \
  --in-geom path/cell_boundaries.csv.gz \
  --g-icol-id 0 \
  --g-icol-x 1 \
  --g-icol-y 2 \
  --pmtiles-zoom 18 \
  --out path/cells.z18.pmtiles
```

Options:

- `--in-geom` points to the polygon geometry file
- `--id-col` is required in generic polygon mode and names the polygon ID column in the factor-probability table
- `--geom-format` can be `auto`, `table`, `geojson`, or `json`; `auto` uses the file extension
- `--geom-id-prop` names the GeoJSON/JSON feature property used as the polygon ID
- `--id-is-u32` tells `punkst` to parse that input ID directly as a `u32` MLT feature ID
- if `--id-is-u32` is not used, `punkst` assigns each polygon an internal feature ID in first-encounter order from the geometry file
- `--keep-org-id` optionally keeps the original input string ID as a regular string property column in the PMTiles output
- `--g-icol-id`, `--g-icol-x`, and `--g-icol-y` set the 0-based table-geometry columns for polygon ID, `x`, and `y`
- `--g-icol-order` is optional; if omitted, the geometry rows are assumed to already be in vertex order
- `--out-sidecar-tsv` optionally writes `polygon_id`, `part_index`, `feature_id`, `center_x`, and `center_y`
- `--cartoscope-boundary` writes the CartoScope boundary schema (`cell_id`, `topK`, `topP`, `K2/P2`, ...); factor IDs in boundary properties are stored as strings for CartoScope compatibility

Geometry file format:

- plain or gzipped TSV/CSV, or GeoJSON/JSON with Polygon/MultiPolygon features
- one row per vertex
- empty lines and lines starting with `#` are ignored
- if the first non-comment line does not contain numeric `x` and `y` values at the selected geometry columns, it is treated as a header and skipped
- later malformed numeric rows are treated as errors

Example geometry file:

```text
cell_id,vertex_x,vertex_y
1,1901.875,2526.4126
1,1901.45,2537.0376
1,1891.12,2537.45
```

Output:

- writes one PMTiles archive at the requested zoom level
- stores the polygon ID in the dedicated MLT feature ID column
- stores `topK`, `topP`, and retained dense factor probabilities as feature properties; in `--cartoscope-boundary` mode it stores nullable `K2/P2...Kn/Pn` instead
- if `--keep-org-id` is used, also stores the original string ID as a regular string property column
- output from this step can be passed to `punkst build-pyramid --polygon`

Boundary behavior:

- by default, polygons that cross tile boundaries are clipped to each tile plus a `--tile-buffer-px` screen-pixel buffer, and may appear in more than one tile
- with `--no-clipping`, polygons are duplicated across touched tiles but kept intact in each copy
- with `--no-duplication`, each polygon is assigned to a single tile and stored there intact
- `--no-duplication` is mainly useful when polygons are much smaller than the tile size

For CartoScope cell-level PMTiles and deployment packaging, see [Deploy punkst results to CartoScope](../workflows/deploy_carto.md#write-cell-level-pmtiles-for-cartoscope).

## Write generic polygon PMTiles from arbitrary properties

`punkst poly2pmtiles-generic` writes a single-zoom polygon PMTiles archive from
an arbitrary property TSV plus polygon geometry. It reuses the same simple
polygon geometry reader and clipping behavior as `poly2pmtiles`, but does not
interpret the input as factor probabilities.

```bash
punkst poly2pmtiles-generic \
  --in-tsv cell_stats.tsv \
  --id-col cell_id \
  --in-geom cell_boundaries.tsv \
  --g-icol-id 0 --g-icol-x 1 --g-icol-y 2 --g-icol-order 3 \
  --string-cols cell_name,cell_type \
  --int-cols cluster \
  --float-cols residual,cosine_sim,entropy \
  --format MVT \
  --pmtiles-zoom 18 \
  --layer-name cell_boundaries \
  --out cell_stats.z18.pmtiles
```

The input TSV must have a header. `--id-col` names the column used to join rows
to geometry records. The same ID is also written as a string property so that
`build-pyramid --polygon-id-col <id-col>` can reconstruct lower zoom levels from
the original geometry source.

Property columns are explicit:

- `--string-cols`, `--int-cols`, and `--float-cols` accept repeated values or comma-separated names.
- Missing values (`""`, `NA`, `NaN`, `nan`, `NULL`, `null`) are written as nullable properties.
- The output supports both `--format MVT` and `--format MLT`; MVT is the browser-oriented choice.

After writing the single-zoom archive, build a pyramid with:

```bash
punkst build-pyramid \
  --polygon-in cell_stats.z18.pmtiles \
  --polygon-source cell_boundaries.tsv \
  --polygon-id-col cell_id \
  --icol-id 0 --icol-x 1 --icol-y 2 --icol-order 3 \
  --min-zoom 10 \
  --out cell_stats.pmtiles
```

## Build PMTiles pyramids

`punkst build-pyramid` builds an MLT or MVT PMTiles pyramid from existing max-zoom PMTiles inputs.

Use `--point-in` for point PMTiles, `--polygon-in` for simple-polygon PMTiles, or both to write one mixed point+polygon pyramid. The older `--point --in ...` and `--polygon --in ...` forms are still accepted for compatibility.

### Point-only pyramids

```bash
punkst build-pyramid \
  --point-in path/pixel.z18.pmtiles \
  --min-zoom 10 \
  --max-tile-bytes 5000000 \
  --max-tile-features 50000 \
  --scale-factor-compression 10 \
  --threads 4 \
  --out path/pixel.pyramid.pmtiles
```

### Simple-polygon pyramids

`punkst build-pyramid --polygon` builds lower zoom levels for simple-polygon MLT or MVT PMTiles.

Current support is limited to:

- simple polygons only
- no holes
- no multipolygon in the internal pyramid representation
- polygon inputs that carry one unique polygon ID per feature

```bash
punkst build-pyramid \
  --polygon-in path/hex.z18.pmtiles \
  --min-zoom 10 \
  --polygon-priority area \
  --max-tile-bytes 5000000 \
  --max-tile-features 50000 \
  --scale-factor-compression 10 \
  --threads 4 \
  --out path/hex.pyramid.pmtiles
```

- `--polygon-priority` chooses how polygons are retained when down-sampling is needed: `random`, `area` (default, in decreasing order)

### Mixed point and polygon pyramids

Provide both input flags to build a single PMTiles archive with a point layer and a polygon layer:

```bash
punkst build-pyramid \
  --point-in path/pixel.z18.pmtiles \
  --polygon-in path/hex.z18.pmtiles \
  --min-zoom 10 \
  --max-tile-bytes 5000000 \
  --max-tile-features 50000 \
  --threads 4 \
  --out path/mixed.pyramid.pmtiles
```

The point and polygon inputs must use the same PMTiles vector format (`MLT` or `MVT`), gzip tile compression, and max zoom. The polygon input follows the same simple-polygon metadata requirements as polygon-only pyramid building.

If a single input PMTiles already contains both point and polygon layers, use the explicit compatibility form:

```bash
punkst build-pyramid --mixed \
  --in path/mixed.z18.pmtiles \
  --min-zoom 10 \
  --out path/mixed.pyramid.pmtiles
```

Behavior:

- by default, if the input PMTiles has an MLT feature ID column, that is used as the polygon ID
- `--polygon-id-col` is optional and acts as a hard override when you want to use a regular property column instead of the MLT feature ID column
- if neither an MLT feature ID column nor `--polygon-id-col` is available, the command reports an error
- for PMTiles written by `punkst poly2pmtiles` in hex mode, which contains only hexagons, a shortcut is taken by using the stored hexagonal grid coordinates to build the geometry instead of recovering it from the encoded geometry
- for generic polygon inputs, uses the finest input zoom level to recover one canonical polygon per ID before building parent levels
- if `--polygon-source` is provided, that file is used as the geometry source override
- by default, parent tiles use clipped polygons that may appear in more than one tile, with clipping buffered by `--tile-buffer-px` screen pixels
- `--tile-buffer-px` overrides that buffer in `--polygon` mode; if omitted, the value stored in the input archive metadata is used
- with `--no-clipping`, parent tiles keep duplicated polygons intact instead of clipping them to tile boundaries
- with `--no-duplication`, each polygon is kept intact and written to only one tile per zoom level
- `--no-clipping` and `--no-duplication` are mutually exclusive

Limitations for the generic polygon mode:

- supports only simple polygons without holes
- when the truth `--polygon-source` is not provided and polygons are recovered from the max-zoom geometry, we may need to repair polygons that are clipped and duplicated across tile boundaries. If a polygon becomes multipolygons or acquires holes after that repair, it is ignored


Optional polygon source file:

`--polygon-source` optionally specifies a polygon-vertex table to override geometry recovery from the finest input zoom. Plain or gzipped TSV/CSV files are supported.

By default, coordinates read from `--polygon-source` are multiplied by the input archive's `coord_scale` metadata before EPSG:3857 tiling so they match the packaged PMTiles coordinate space. Use `--polygon-source-coord-scale` to override that default when the source table is already pre-scaled or uses a different scale.

Required fields (specified by 0-based column index):
- polygon ID `--icol-id` (default: `0`)
- `x` coordinate `--icol-x` (default: `1`)
- `y` coordinate `--icol-y` (default: `2`)

Optional field: vertex order `--icol-order`. If provided, it is used to assemble each polygon ring, otherwise, rows in the input file are assumed to already be in polygon-vertex order.
- `--polygon-source-coord-scale` optionally overrides the scale applied to `--polygon-source` coordinates before EPSG:3857 tiling

(Empty lines and lines starting with `#` are ignored; if the first non-comment line does not have numeric `x` and `y` values at `--icol-x` and `--icol-y`, it is treated as a header line and skipped; after that first line, malformed numeric values are treated as errors)

Example:

```text
ID,x,y
cell_1,10.0,20.0
cell_1,12.0,20.0
cell_1,12.0,22.0
cell_1,10.0,22.0
```

### Shared requirements and options

Requirements:

- `--in` (or `--in-data`, used equivalently) must point to the input PMTiles archive
- `--out` must be a concrete PMTiles file path
- choose point, polygon, or mixed mode. Use `--point-in`, `--polygon-in`, or both for the explicit form; the older compatibility form uses `--point`, `--polygon`, or `--mixed` with `--in`

Main options:

- `--min-zoom` sets the coarsest zoom that should exist in the output archive
- `--max-tile-bytes` sets a target upper bound on compressed tile size (default: 5MB)
- `--max-tile-features` sets a target upper bound on the number of features per tile (default: 50K)
- `--scale-factor-compression` controls how aggressively features are kept before the final tile-size check (default: 10)
- `--threads` controls parallel processing
- `--polygon-id-col` is an optional hard override for polygon ID lookup in `--polygon` mode
- `--polygon-priority` sets polygon retention mode for `--polygon`
- `--polygon-source` and `--icol-*` options are used only for `--polygon` in generic polygon mode
- `--tile-buffer-px` sets the screen-pixel clip buffer for the default clipped polygon mode
- `--no-clipping` switches polygon output to intact duplicated polygons without tile-boundary clipping
- `--no-duplication` switches polygon output from clipped/duplicated tiles to single-tile intact storage

Input PMTiles handling:

- if the input already contains zoom levels down to or below `--min-zoom`, the command exits directly
- if the input already contains some lower zoom levels but not enough, those existing levels are preserved byte-for-byte and only the missing coarser levels are built
