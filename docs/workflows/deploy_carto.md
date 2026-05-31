# Deploy punkst results to CartoScope

`punkst deploy-cartoscope` packages completed punkst workflow output into a CartoScope deployment directory.

Run it after the standard punkst [workflow](./index.md) has finished and one or more factor models have been fitted.

## Scope
The command does not run fitting or decoding. It starts from existing tiled transcripts, model outputs, pixel decode files, then writes the PMTiles and metadata files CartoScope needs.

The deployment contains transcript PMTiles, gene-bin PMTiles, merged raw-pixel factor annotations, hexagon factor PMTiles, optional pixel-raster factor PMTiles, sidecar tables, `ficture_assets.json`, and `catalog.yaml`. It does not package imaging, basemaps, UMAPs, cells, or boundaries.

## Basic Usage

```bash
punkst deploy-cartoscope \
  --config /path/to/punkst/output/config.json \
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
  --config /path/to/punkst/output/config.json \
  --out-dir /path/to/cartoscope_deploy \
  --id sample-id \
  --title "Sample title" \
  --pmtiles-format MVT \
  --threads 4 \
  --overwrite
```
(Caution: if `--overwrite` is not set, files with the intended output names that exist in the output directory will be reused, even if they might be stale)

## Input Modes

### Standard Workflow Config

Use `--config` for output produced by the standard workflow template (see [Quickstart](index.md) and `punkst/examples/basic/`).

For each model prefix, the workflow output should contain files like:

```text
<model-prefix>.results.tsv
<model-prefix>.model.tsv
<model-prefix>.color.rgb.tsv
<model-prefix>.pixel.bin
<model-prefix>.pixel.index
<model-prefix>.pixel.pseudobulk.tsv
<model-prefix>.pixel.de_bulk.tsv
```

If `workflow.pixel_decode_mode` is set, deployment uses the matching pixel-output suffix:

```text
pixel             -> <model-prefix>.pixel.*
feature_pixel     -> <model-prefix>.sf_pixel.*
single_molecule   -> <model-prefix>.sgl_mol.*
```

Default deployment model IDs use names such as `h12-k12`, `h12-k12-sf-pixel`, and `h12-k12-sgl-mol`.

Deploy only selected models with `--model-prefix`, e.g. `--model-prefix <model-prefix> hex_12.k24`

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
      "pixel_decode_mode": "pixel",
      "pixel_png": "/path/to/hex_12.k12.pixel.png",
      "pseudobulk_tsv": "/path/to/hex_12.k12.pixel.pseudobulk.tsv",
      "de_tsv": "/path/to/hex_12.k12.pixel.de_bulk.tsv"
    }
  ]
}
```

Relative paths are resolved relative to the JSON file. `pixel_decode_mode` is optional; when omitted, deployment infers the mode from `<pixel_prefix>.index`. `pixel_png` is optional. When `pixel_png` is omitted, the factor is deployed without `pmtiles.raster`; raw-pixel vector rendering still uses the annotated transcript PMTiles.

If the same fitted model is decoded in multiple pixel modes, list those outputs as separate model entries with different `id` values. The entries can reuse the same `results_tsv`, `model_tsv`, and `color_rgb_tsv` while pointing to different pixel prefixes, pseudobulk tables, DE tables, and optional PNGs.

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
  - id: h12-k12
    model_id: h12-k12
    decode_id: h12-k12-pixel
    raw_pixel_col: h12-k12-pixel
    de: h12-k12-bulk-de.tsv
    info: h12-k12-info.tsv
    rgb: h12-k12-rgb.tsv
    pmtiles:
      hex: h12-k12.pmtiles
      raster: h12-k12-pixel-raster.pmtiles
      raw_pixel: genes_all.pmtiles
```

`pmtiles.raw_pixel` points to `genes_all.pmtiles`, which contains transcript points annotated with per-model pixel factor columns such as `h12-k12-pixel_K1` and `h12-k12-pixel_P1`.

Explicit JSON model IDs in output filenames and catalog entries are normalized by replacing `_` and `.` with `-`.

## PMTiles and Zooms

(See more about PMTiles utilities in [Modules](../modules/pmtiles.md))

Transcript PMTiles are written at `--point-max-zoom`, then pyramid levels are built down to `--point-min-zoom`.

Hexagon PMTiles are written at `--polygon-max-zoom`, then pyramid levels are built down to `--polygon-min-zoom`.

Gene-bin PMTiles are controlled by:

```bash
--n-gene-bins 50
```
