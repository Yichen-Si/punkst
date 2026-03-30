# pmtiles

**PMTiles utilities in `punkst` currently cover:**

- [Export point-only PMTiles as TSV](#export-pmtiles-as-tsv)
- [Write single-level polygon-only PMTiles](#write-single-zoom-polygon-pmtiles)
- [Build PMTiles pyramids](#build-mlt-pmtiles-pyramids)

We only support PMTiles with the [MLT format](https://maplibre.org/maplibre-tile-spec/).

For writing point/pixel-only PMTiles see [tile-op](tileop.md)

## Export PMTiles as TSV

`punkst export-pmtiles` reads an MLT-backed PMTiles archive that contains **only point data** and exports it back to a plain TSV plus `.index` that `tile-op` can read again.

```bash
punkst export-pmtiles \
  --in path/prefix.pmtiles \
  --tile-size 500 \
  --out path/prefix
```

Requirements and behavior:

- `--in` or `--in-data` must point to the PMTiles archive with **MLT** format
- `--out` specify the output prefix. The output includes `path/prefix.tsv` and `path/prefix.index`
- `--tile-size` is required and defines the tile size for the output file (in the original units)
- the data is read from the archive's max-zoom level
- output columns are `x`, `y`, optional `z`, then the decoded PMTiles schema columns
- missing `K` and `P` values are rendered as `-1` and `0`; other nullable fields are rendered as `NA`


Region filters are supported in this mode:

- `--xmin`, `--xmax`, `--ymin`, `--ymax`
- `--extract-region-geojson`
- `--zmin`, `--zmax`

## Write single-zoom polygon PMTiles

`punkst hex2pmtiles` writes a **polygon-only MLT PMTiles with a single zoom level** from a factor-probability table (output from [`punkst topic-model`](lda4hex.md)).

It supports two input styles:

- hexagon mode: build polygon geometry from hex centers
- generic polygon mode: link factor-probability rows to polygon geometry by polygon ID

### Hexagon mode

Use this when the input table contains `x` and `y` coordinates for hex centers.

```bash
punkst hex2pmtiles \
  --in-tsv path/model.results.tsv.gz \
  --hex-grid-dist 12 \
  --pmtiles-zoom 18 \
  --out path/model.hex.z18.pmtiles
```

Options:

- `--hex-grid-dist` (required) specifies the distance between adjacent grid points on the hexagonal grid
- `--pmtiles-zoom` (required) sets the output zoom level
- `--x-col`, `--y-col`, `--topk-col`, and `--topp-col` set column names in the input factor table for the hex center coordinates (default: `x` and `y`), the top factor ID (default: `topK`), and the top factor probability (default: `topP`)
- `--prob-thres` keeps only factor probabilities above the threshold as nullable properties (default: `1e-4`)
- `--coord-scale` scales the input coordinates before Web Mercator tiling (default: `1`, no scaling)
- `--no-duplication` stores each polygon intact in exactly one tile at the requested zoom instead of clipping and duplicating it across tile boundaries

### Generic simple-polygon mode

Use this when the factor-probability table contains a polygon ID column and polygon geometry is provided in a separate file. For example, if you ran `punkst topic-model` on segmented cells and you want to create a PMTiles archive with cell boundaries as geometry.

```bash
punkst hex2pmtiles \
  --in-tsv path/cells.results.tsv \
  --id-col cell_id \
  --in-geom path/cell_boundaries.csv.gz \
  --icol-id-geom 0 \
  --icol-x-geom 1 \
  --icol-y-geom 2 \
  --pmtiles-zoom 18 \
  --out path/cells.z18.pmtiles
```

Options:

- `--in-geom` points to the polygon geometry file
- `--id-col` is required in generic polygon mode and names the polygon ID column in the factor-probability table
- `--id-is-u32` tells `punkst` to parse that input ID directly as a `u32` MLT feature ID
- if `--id-is-u32` is not used we assign each polygon an internal `u32` feature ID in first-encounter order from the geometry file and write a sidecar `*.idmap.tsv` with the mapping from the input string ID to the assigned integer ID
- `--keep-org-id` optionally keeps the original input string ID as a regular string property column in the PMTiles output
- `--icol-id-geom`, `--icol-x-geom`, and `--icol-y-geom` set the 0-based geometry-file columns for polygon ID, `x`, and `y`
- `--icol-order-geom` is optional; if omitted, the geometry rows are assumed to already be in vertex order
- `--no-duplication` stores each polygon intact in exactly one tile at the requested zoom instead of clipping and duplicating it across tile boundaries

Geometry file format:

- plain or gzipped TSV/CSV
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
- stores `topK`, `topP`, and retained factor probabilities as feature properties
- if `--keep-org-id` is used, also stores the original string ID as a regular string property column
- output from this step can be passed to `punkst build-pyramid --polygon`

Boundary behavior:

- by default, polygons that cross tile boundaries are clipped and may appear in more than one tile
- with `--no-duplication`, each polygon is assigned to a single tile and stored there intact
- `--no-duplication` is mainly useful when polygons are much smaller than the tile size

## Build MLT PMTiles pyramids

`punkst build-pyramid` builds a MLT PMTiles pyramid from an existing MLT PMTiles input.

Currently it only support point-only and simple-polygon-only PMTiles. (This is the intended next step after writing a max-zoom archive with `punkst tile-op` or `punkst hex2pmtiles`)
Exactly one of `--point` or `--polygon` must be specified.

### Point-only pyramids

```bash
punkst build-pyramid --point \
  --in path/pixel.z18.pmtiles \
  --min-zoom 10 \
  --max-tile-bytes 5000000 \
  --max-tile-features 50000 \
  --scale-factor-compression 10 \
  --threads 4 \
  --out path/pixel.pyramid.pmtiles
```

### Simple-polygon pyramids

`punkst build-pyramid --polygon` builds lower zoom levels for simple-polygon MLT PMTiles.

Current support is limited to:

- simple polygons only
- no holes
- no multipolygon
- polygon inputs that carry one unique polygon ID per feature

```bash
punkst build-pyramid --polygon \
  --in path/hex.z18.pmtiles \
  --min-zoom 10 \
  --polygon-priority area \
  --max-tile-bytes 5000000 \
  --max-tile-features 50000 \
  --scale-factor-compression 10 \
  --threads 4 \
  --out path/hex.pyramid.pmtiles
```

- `--polygon-priority` chooses how polygons are retained when down-sampling is needed: `random`, `area` (default, in decreasing order)

Behavior:

- by default, if the input PMTiles has an MLT feature ID column, that is used as the polygon ID
- `--polygon-id-col` is optional and acts as a hard override when you want to use a regular property column instead of the MLT feature ID column
- if neither an MLT feature ID column nor `--polygon-id-col` is available, the command reports an error
- for PMTiles written by `punkst hex2pmtiles`, which contains only hexagons, a shortcut is taken by using the stored hexagonal grid coordinates to build the geometry instead of recovering it from the encoded geometry
- for generic polygon inputs, uses the finest input zoom level to recover one canonical polygon per ID before building parent levels
- if `--polygon-source` is provided, that file is used as the geometry source override
- by default, parent tiles use clipped polygons that may appear in more than one tile
- with `--no-duplication`, each polygon is kept intact and written to only one tile per zoom level

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
- exactly one of `--point` or `--polygon` must be specified

Main options:

- `--min-zoom` sets the coarsest zoom that should exist in the output archive
- `--max-tile-bytes` sets a target upper bound on compressed tile size (default: 5MB)
- `--max-tile-features` sets a target upper bound on the number of features per tile (default: 50K)
- `--scale-factor-compression` controls how aggressively features are kept before the final tile-size check (default: 10)
- `--threads` controls parallel processing
- `--polygon-id-col` is an optional hard override for polygon ID lookup in `--polygon` mode
- `--polygon-priority` sets polygon retention mode for `--polygon`
- `--polygon-source` and `--icol-*` options are used only for `--polygon` in generic polygon mode
- `--no-duplication` switches polygon output from clipped/duplicated tiles to single-tile intact storage

Input PMTiles handling:

- if the input already contains zoom levels down to or below `--min-zoom`, the command exits directly
- if the input already contains some lower zoom levels but not enough, those existing levels are preserved byte-for-byte and only the missing coarser levels are built
