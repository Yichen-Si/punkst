# Multi-Sample Analysis Utilities

There are two main utilities for handling multiple transcriptomics inputs to help multi-sample analysis.

`multisample-prepare` processes raw data from multiple samples in a unified way. The output are ready for model training and later sample-specific pixel level projection. You also have the option to run only the first or the second step of the pipeline, see below for details.

`merge-units` merges multiple binned datasets, for example single cells or hexagons, in our customized sparse matrix format to a single dataset while harmonizing the sample-specific feature lists.

(`pixel-decode` also allows processing multiple samples using the same model and parameters (see option `--sample-list` in [`pixel-decode`](../modules/pixel-decode.md)) )

---

## Processing multiple samples from raw data

The `multisample-prepare` command processes multiple raw spatial transcriptomics datasets into a merged hexagonal binned data suitable for joint model training (by `punkst topic-model`) and sample-specific tiled pixel level data for pixel level projection.

It runs [`pts2tiles`](../modules/pts2tiles.md) and [`tiles2hex`](../modules/tiles2hex.md) for each sample then merge the binned level data.

(if neither `--hex-grid-dist` nor `--hex-size` is provided, it only runs `pts2tiles`; if `--tiles2hex-only` is set (see the third section below) it only runs `tiles2hex` and merges the output files.)

### Usage

```bash
punkst multisample-prepare --in-tsv-list input_file_list.tsv \
    --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 --skip 1 \
    --tile-size 500 \
    --min-total-count-per-sample 100 \
    --hex-grid-dist 12 --min-count 10 \
    --out-dir ./out --out-joint-pref merged \
    --temp-dir ./tmp --threads ${threads}
```

### Input File List (`--in-tsv-list`)

The pipeline requires an input TSV file that lists the information for each sample to be processed.
Each line should contain two tab-separated columns:
1.  A unique Sample ID.
2.  The path to the raw transcript file for that sample.

The raw transcript file should be in the format expected by [`pts2tiles`](../modules/pts2tiles.md).
The raw transcript file should be in the format expected by [`pts2tiles`](../modules/pts2tiles.md).

**Example `input_file_list.tsv`:**
```tsv
sample_A	/path/to/sample_A_transcripts.tsv
sample_B	/path/to/sample_B_transcripts.tsv
```


### Key Options

(Other options are available, see [`pts2tiles`](../modules/pts2tiles.md) and [`tiles2hex`](../modules/tiles2hex.md) for details.)
(Other options are available, see [`pts2tiles`](../modules/pts2tiles.md) and [`tiles2hex`](../modules/tiles2hex.md) for details.)

`--in-tsv-list <file>`: (Required) Path to the input TSV file describing the samples.

`--min-total-count-per-sample <int>`: The minimum sample-specific total count a feature must have (across all samples) to be included in the merged file. Default: 1. Setting it to 0 to use the union of features across samples.

`--exclude-feature-regex`: Regular expression ([modified ECMAScript grammar](https://en.cppreference.com/w/cpp/regex/ecmascript)) to exclude features matching this pattern. Default: exclude no feature.

`--include-feature-regex`: Regular expression to include only features matching this pattern. Default: include all features. (e.g. to exclude features that contain "Unassigned" or "NegControl" as substrings in any part of the feature name, you can use `--exclude-feature-regex ".*(Unassigned|NegControl).*"`)

`--threads <int>`: Number of threads to use. \[Default: 1]

`--out-dir <dir>`: (Required) The base output directory where all results will be stored. Merged outputs will be placed directly under this directory, sample-specific outputs will be in subdirectories named `samples/[sample_id]/`.

`--out-joint-pref <prefix>`: (Required) A prefix for all merged output (the merged hexagon file and merged feature list). For example, the

`--temp-dir <dir>`: (Required) A directory for storing temporary files (will be created if it doesn't exist).

`--overwrite`: If set, overwrite existing sample-specific output files.

**`pts2tiles` options:**

`--icol-x <int>`, `--icol-y <int>`, `--icol-feature <int>`, `--icol-int <int>`: (Required) 0-based column indices for X/Y coordinates, the feature name, and the count/value.

`--skip`: If your input file has a header, use `--skip 1` to skip the first (or more) lines.

`--tile-size <int>`: (Required) The size of the square tiles for pre-processing. Should be big enough, say 500 microns.

**`tiles2hex` options:**

`--hex-grid-dist <float>`: The center-to-center distance for the hexagonal grid. Alternatively, provide `--hex-size <float>`, side length of the hexagons (exactly one of the two options must be provided). Multiple values can be provided, separated by spaces.

`--min-count <int>`: The minimum total count for a hexagon (unit) to be included in the output.

### Output Files

All outputs are under the specified `--out-dir`

**In the main output directory (`--out-dir`):**

`[--out-joint-pref].persample_file_list.tsv`: A list of paths to the tiled pixel level files for each sample. These are the input for `pixel-decode`.

`[--out-joint-pref].union_features.tsv`: A list of all features found in any of the samples, with total and sample-specific counts.

`[--out-joint-pref].features.tsv`: The final list of features used for the merged output.

`[--out-joint-pref].hex_[dist].txt` and `.json`: The final merged hexagon data and its corresponding metadata file, ready for `lda4hex`.

**In per-sample subdirectories (`--out-dir/samples/[sample_id]/`):**

Intermediate tiled transcript files (`.tiled.tsv`, `.tiled.index`).

Per-sample feature counts (`.tiled.features.tsv`).

Per-sample randomized hexagon data (`.hex_[dist].txt`, `.hex_[dist].json`).

---

## Merge hexagon units from pre-processed samples

The `merge-units` command merges multiple bin level data in the format of the output from `punkst tiles2hex`.

The input files can have different extra information, as long as the metadata (`.json`) are recognized and includes a key `offset_data` indicating the starting index (0-based) of the sparsely coded count data.

(In each row, tokens are separate by tabs. Starting from the index specified by `offset_data`, each row contains two integers for the number of unique features and the total count of all features, followed by feature_index and count (separated by a single space) pairs. See [`tiles2hex`](../modules/tiles2hex.md) for more details.)
(In each row, tokens are separate by tabs. Starting from the index specified by `offset_data`, each row contains two integers for the number of unique features and the total count of all features, followed by feature_index and count (separated by a single space) pairs. See [`tiles2hex`](../modules/tiles2hex.md) for more details.)


### Usage

This example merges two pre-processed samples.

Two optional input specifications are demonstrated:
It tells the tool to either use the existing random keys or generate new random keys based on the input data (`-2` on the 5-th column) and to carry over the data from column index (0-based) `4` of sample 1 and column `3` from sample 2 into the new "info" column.

```bash
# Create the input specification
input_list="input.tsv"
echo -e "1\t./1/1.tiled.features.tsv\t./1/1.hex_12.txt\t./1/1.hex_12.json\t-2\t4" > ${input_list}
echo -e "2\t./2/2.tiled.features.tsv\t./2/2.hex_12.txt\t./2/2.hex_12.json\t-2\t3" >> ${input_list}

# Run the merge command
punkst merge-units \
    --in-list ${input_list} \
    --out-pref ./merged.hex_12 \
    --min-total-count-per-sample 100 \
    --temp-dir ./tmp --threads 4
```

### Input File (`--in-list`)

The command requires a TSV file specifying the input for each sample. Each line must contain at least four columns:

1.  **Sample ID**: A unique identifier for the sample.

2.  **Features Path**: Path to the sample-specific feature file. (TSV with feature name and count, like that created by `pts2tiles`. We only read the first two columns; lines where the second token is not a non-negative integer are ignored.)

3.  **Hexagon Data Path**: Path to the sample's hexagon data file (`.txt`).

4.  **Hexagon Metadata Path**: Path to the sample's hexagon metadata file (`.json`).

5.  **Random Key Column Index (Optional)**: An integer specifying how to handle the random key for each unit.

    - `>= 0`: The column index in the input hexagon file to use as the random key for shuffling the merged output.

    - `-1`: Generate a new random key for each unit.

    - `-2`: Try to find the key column from the metadata (`.json`) file first. (Default)

6.  **Info Columns (Optional)**: A comma-delimited string of column indices (e.g., `2,5,6`) from the input hexagon file to carry over into a single "info" column in the merged output. Each sample can have different info columns. Put an "." to indicate no info columns to carry over for that sample.

**Example `input.tsv`:**
```tsv
1	/path/to/1/1.tiled.features.tsv	/path/to/1/1.hex_12.txt	/path/to/1/1.hex_12.json	0	4
2	/path/to/2/2.tiled.features.tsv	/path/to/2/2.hex_12.txt	/path/to/2/2.hex_12.json	-2	3
```

### Options

`--in-list <file>`: (Required) Path to the input TSV file listing the pre-processed samples.

`--out-pref <prefix>`: (Required) Prefix for all output files (e.g., `/path/to/output/merged`).

`--temp-dir <dir>`: (Required) A directory for storing temporary files (will try to create it if it doesn't exist).

`--min-total-count-per-sample <int>`: Minimum per-sample count a feature must have across all samples to be included in the final merged feature set. (Default: 1. Set to 0 to use the union of features).

`--min-count-per-unit <int>`: Minimum total count a unit/hexagon must have after feature filtering to be included in the merged output. \[Default: 1]

`--threads <int>`: Number of threads. \[Default: 1]

### Output Files

`[--out-pref].txt` and `.json`: The merged hexagon data file and its corresponding metadata, ready for `lda4hex`.

`[--out-pref].features.tsv`: The list of features and their total counts in the merged dataset.

`[--out-pref].union_features.tsv`: A list of all features found across all samples and their per-sample counts.

---

## Generate sample-specific and merged hexagons

The `multisample-prepare` command has another mode to only perform the second step: run [`tiles2hex`](../modules/tiles2hex.md) for each sample then merge the binned level data. This mode is activated by `--tiles2hex-only`, and a different input should be provided in `--in-tsv-list`.
Most likely use case is when you have already run `pts2tiles` for each sample separately.

### Usage

```bash
punkst multisample-prepare --in-tsv-list input_file_list.tsv \
    --icol-x 0 --icol-y 1 --icol-feature 2 --icol-int 3 \
    --tiles2hex-only \
    --min-total-count-per-sample 100 \
    --hex-grid-dist 12 --min-count-per-unit 10 \
    --out-dir ./out --out-joint-pref merged \
    --temp-dir ./tmp --threads ${threads} \
```

### Input File List (`--in-tsv-list`)

The pipeline requires an input TSV file that lists the information for each sample to be processed.
Each line should contain two tab-separated columns:
1.  A unique Sample ID.
2.  The path to the tiled transcript file for that sample.
3.  The path to the corresponding index file.
4.  The path to the per-sample feature count file.

The three input files per sample should be in the same formats as those output by [`pts2tiles`](../modules/pts2tiles.md).

**Example `input_file_list.tsv`:**
```tsv
sample_A	/path/to/sample_A_transcripts.tiled.tsv /path/to/sample_A_transcripts.tiled.index   /path/to/sample_A.features.tsv
sample_B	/path/to/sample_B_transcripts.tiled.tsv /path/to/sample_B_transcripts.tiled.index   /path/to/sample_B.features.tsv
```
