# Deploy punkst results to CartoScope

`punkst deploy-cartoscope` packages completed punkst workflow output into a CartoScope deployment directory.

Run it after the standard punkst [workflow](./index.md) has finished and one or more factor models have been fitted.

## Scope
The command does not run fitting or decoding. It starts from existing tiled transcripts, model outputs, pixel decode files, then writes the PMTiles and metadata files CartoScope needs.

The deployment contains transcript PMTiles, gene-bin PMTiles, merged raw-pixel factor annotations, hexagon factor PMTiles, optional pixel-raster factor PMTiles, sidecar tables, `ficture_assets.json`, and `catalog.yaml`. It does not package imaging, basemaps, UMAPs, cells, or boundaries.

## Basic Usage

```bash
punkst deploy-cartoscope \
  --in-dir /path/to/punkst/output \
  --out-dir /path/to/cartoscope_deploy \
  --id sample-id \
  --title "Sample title" \
  --pmtiles-format MVT \
  --threads 4
```

`--pmtiles-format` is required and must be either `MLT` or `MVT`.

Existing output files are reused by default. Add `--overwrite` to regenerate files that are already present:

```bash
punkst deploy-cartoscope \
  --in-dir /path/to/punkst/output \
  --out-dir /path/to/cartoscope_deploy \
  --id sample-id \
  --title "Sample title" \
  --pmtiles-format MVT \
  --threads 4 \
  --overwrite
```
(Caution: if `--overwrite` is not set, files with the intended output names that exist in the output directory will be reused, even if they might be stale)

## Input Modes

### Standard Workflow Directory

Use `--in-dir` as above for output produced by the standard workflow template (see [Quickstart](index.md) and `punkst/examples/basic/`)

By default, the command reads `/path/to/out/config.json` (override the config path with `--config`) and discovers model prefixes from `workflow.hexgrids` and `workflow.topics`, for example:

```text
hex_12.k12
hex_12.k24
```

For each model prefix, the workflow output should contain files like:

```text
hex_12.k12.results.tsv
hex_12.k12.model.tsv
hex_12.k12.color.rgb.tsv
hex_12.k12.pixel.bin
hex_12.k12.pixel.index
hex_12.k12.pixel.pseudobulk.tsv
hex_12.k12.pixel.de_bulk.tsv
hex_12.k12.pixel.info.tsv
```

Deploy only selected models with `--model-prefix`, e.g. `--model-prefix hex_12.k12 hex_12.k24`

### Explicit Input JSON

Use `--input-json` when inputs are not arranged like the standard workflow output.

```bash
punkst deploy-cartoscope \
  --input-json deploy_inputs.json \
  --out-dir /path/to/deploy \
  --id sample-id \
  --title "Sample title" \
  --pmtiles-format MVT
```

Example:

```json
{
  "transcripts": {
    "tiled_prefix": "/path/to/transcripts.tiled",
    "feature_count_tsv": "/path/to/transcripts.tiled.features.tsv",
    "icol_x": 0,
    "icol_y": 1,
    "icol_feature": 2,
    "icol_count": 3
  },
  "models": [
    {
      "id": "hex-12-k12",
      "hex_grid_dist": 12,
      "results_tsv": "/path/to/hex_12.k12.results.tsv",
      "model_tsv": "/path/to/hex_12.k12.model.tsv",
      "color_rgb_tsv": "/path/to/hex_12.k12.color.rgb.tsv",
      "pixel_prefix": "/path/to/hex_12.k12.pixel",
      "pixel_png": "/path/to/hex_12.k12.pixel.png",
      "pseudobulk_tsv": "/path/to/hex_12.k12.pixel.pseudobulk.tsv",
      "de_tsv": "/path/to/hex_12.k12.pixel.de_bulk.tsv",
      "info_tsv": "/path/to/hex_12.k12.pixel.info.tsv"
    }
  ]
}
```

Relative paths are resolved relative to the JSON file. `info_tsv` and `pixel_png` are optional. When `pixel_png` is omitted, the factor is deployed without `pmtiles.raster`; raw-pixel vector rendering still uses the annotated transcript PMTiles.

## Output

The output directory contains:

```text
catalog.yaml
ficture_assets.json
genes_all.pmtiles
genes_bin*.pmtiles
genes_bin_counts.json
genes_pmtiles_index.tsv
<model-id>.pmtiles
<model-id>-pixel-raster.pmtiles
<model-id>-bulk-de.tsv
<model-id>-info.tsv
<model-id>-model.tsv
<model-id>-pseudobulk.tsv.gz
<model-id>-rgb.tsv
```

`<model-id>-pixel-raster.pmtiles` is written only when a pixel PNG is available. Raster bounds are read from the transcript coordinate range file produced by `pts2tiles`; if that file is absent, deployment falls back to bounds stored in the transcript or pixel index headers.

The catalog registers each factor with the available PMTiles:

```yaml
assets:
  sge:
    all: genes_all.pmtiles
    bins:
    - genes_bin1.pmtiles
    counts: genes_bin_counts.json
  factors:
  - id: hex-12-k12
    model_id: hex-12-k12
    decode_id: hex-12-k12-pixel
    raw_pixel_col: hex-12-k12-pixel
    de: hex-12-k12-bulk-de.tsv
    info: hex-12-k12-info.tsv
    rgb: hex-12-k12-rgb.tsv
    pmtiles:
      hex: hex-12-k12.pmtiles
      raster: hex-12-k12-pixel-raster.pmtiles
      raw_pixel: genes_all.pmtiles
```

`pmtiles.raw_pixel` points to `genes_all.pmtiles`, which contains transcript points annotated with per-model pixel factor columns such as `hex-12-k12-pixel_K1` and `hex-12-k12-pixel_P1`.

Model IDs in output filenames and catalog entries are normalized by replacing `_` and `.` with `-`.

## PMTiles and Zooms

(See more about PMTiles utilities in [Modules](../modules/pmtiles.md))

Transcript PMTiles are written at `--point-max-zoom`, then pyramid levels are built down to `--point-min-zoom`.

Hexagon PMTiles are written at `--polygon-max-zoom`, then pyramid levels are built down to `--polygon-min-zoom`.

Gene-bin PMTiles are controlled by:

```bash
--n-gene-bins 50
```
