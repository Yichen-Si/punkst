# pts2tiles

### Group points into square tiles for faster downstream access

`pts2tiles` rewrites a delimited text point file so records are grouped into non-overlapping square tiles in `(x, y)` space. It writes a tiled `.tsv` output file plus a binary index storing byte offsets for each tile to help faster region query.

The row order is preserved only at the tile level. Records inside a tile may be arbitrary.

Example usage:

```bash
punkst pts2tiles \
  --in-tsv ${path}/transcripts.tsv \
  --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
  --tile-size 500 \
  --temp-dir ${tmpdir} \
  --out-prefix ${path}/transcripts.tiled \
  --threads ${threads}
```

Example with per-axis scaling:

```bash
punkst pts2tiles \
  --in-tsv ${path}/transcripts.tsv \
  --icol-x 0 --icol-y 1 --icol-z 2 \
  --tile-size 500 \
  --scale-x 0.108 --scale-y 0.108 --scale-z 1 \
  --temp-dir ${tmpdir} \
  --out-prefix ${path}/transcripts.tiled
```

Example with CSV input:

```bash
punkst pts2tiles \
  --in-tsv ${path}/transcripts.csv.gz \
  --csv \
  --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
  --tile-size 500 \
  --temp-dir ${tmpdir} \
  --out-prefix ${path}/transcripts.tiled
```

### Accepted Input Forms

`pts2tiles` currently accepts these input forms through `--in-tsv`:

- A plain text file on disk.
- A gzipped text file whose path ends with `.gz`.
- Standard input, using `--in-tsv -`.

Delimiter handling:

- By default, input is parsed as tab-delimited text.
- With `--csv`, input is parsed as comma-delimited text.
- Output is always written as tab-delimited `.tsv`, even when the input is CSV.

Current parser behavior:

- The parser splits each line on a single-character delimiter only.
- It does not implement full RFC-style CSV quoting or escaping.
- In practice, `--csv` should be used only for simple comma-delimited records where commas do not appear inside quoted fields.

Header and metadata handling:

Lines starting with `#` are automatically treated as metadata/header and copied directly to the output regardless of `--skip`, the following options are mainly for lines that do not start with `#` but should be treated as metadata/header.

- `--skip N` skips the first `N` lines from parsing.
- Skipped lines are copied to the output as metadata comments.
- `--skip-last-is-header` marks the last skipped line as the header row and rewrites it to start with a single `#`.
- Earlier skipped lines are normalized to start with at least two `#` characters.

### Main Parameters

`--in-tsv`
: Input text file, gzipped text file, or `-` for stdin.

`--icol-x`, `--icol-y`
: Zero-based column indices for the `x` and `y` coordinates used for tiling.

`--tile-size`
: Side length of each square tile, in the same units as the effective coordinates after optional scaling.

`--out-prefix`
: Prefix for output files.

`--temp-dir`
: Directory used for temporary per-tile files during processing.

### Feature and Value Columns

`--icol-feature`
: Optional zero-based feature column. If provided, `pts2tiles` can also write `prefix.features.tsv`.

`--icol-int`
: Optional zero-based integer column to aggregate per feature. Multiple columns can be provided as `--icol-int 3 4 5`. If not provided, a dummy value of `1` is used assuming each record is one transcript.

Behavior of `prefix.features.tsv`:

- If `--icol-feature` and one or more `--icol-int` columns are provided, each row contains the feature followed by aggregated integer totals for those columns.

### Optional Parameters

`--icol-z`
: Optional zero-based `z` coordinate column. Tiling still uses only `x` and `y`, but if `z` is provided the command also records `zmin`/`zmax` and writes `prefix.z_hist.tsv`.

`--csv`
: Parse the input as comma-delimited text instead of tab-delimited text.

`--skip`
: Number of initial lines to skip from parsing and preserve as metadata in the output. Default: `0`.

`--skip-last-is-header`
: Treat the last skipped line as the header line. Requires `--skip > 0`.

`--tile-buffer`
: Per-thread, per-tile in-memory buffer size in number of records before flushing to disk. Default: `1000`.

`--batch-size`
: Batch size in number of lines when reading from stdin or a gzipped input stream. Only relevant for `--in-tsv -` and inputs ending in `.gz`. Default: `10000`.

`--scale`
: Uniform scaling factor applied to `x`, `y`, and `z` if present, unless axis-specific overrides are supplied. Default: `1`.

`--scale-x`, `--scale-y`, `--scale-z`
: Axis-specific scaling factors. If an axis-specific scale is omitted, it falls back to `--scale`.

`--digits`
: Decimal precision used when rewritten scaled coordinates are written to the output. Default: `2`.

`--threads`
: Number of worker threads. Default: `1`.

`--verbose`
: Verbosity level for logging.

`--debug`
: Enable additional debug output.

### Scaling Notes

- Tile assignment is computed from scaled coordinates.
- If no scaling is requested, coordinates are copied through unchanged.
- If scaling is requested, rewritten coordinate values are emitted using `--digits` decimal places.
- `--scale-z` has an effect only when `--icol-z` is provided.

### Output Files

- `prefix.tsv`: tiled tab-delimited output records.
- `prefix.index`: binary tile index with per-tile byte offsets and bounding boxes.
- `prefix.coord_range.tsv`: global coordinate range containing `xmin`, `xmax`, `ymin`, `ymax`, and also `zmin`, `zmax` if `--icol-z` is provided.
- `prefix.z_hist.tsv`: two-column tab-delimited histogram of `z` using unit-width bins. The unit is the same as the effective `z` after scaling. Written only when `--icol-z` is provided.
- `prefix.features.tsv`: sorted feature summary file. Written only when `--icol-feature` is provided.
