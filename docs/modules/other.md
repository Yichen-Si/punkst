# Other Utilities

This page documents utility commands that are useful in preprocessing or exploratory analysis but are not part of the main topic-model fitting workflow.

## feature-vst

`feature-vst` computes variance-stabilized feature scores and selects highly variable features (HVFs).

`feature-vst` accepts either the custom hex-unit format produced by `tiles2hex` or 10X DGE input in Matrix Market format.

### Basic Usage

```bash
punkst feature-vst \
  --in-data ${path}/hex_12.txt \
  --in-meta ${path}/hex_12.json \
  --min-count-train 20 \
  --min-count-per-feature 100 \
  --n-top 2000 \
  --out-prefix ${path}/feature_vst \
  --threads ${threads}
```

### Input Options

For hex input, provide both:

`--in-data` specifies the hex-unit file.

`--in-meta` specifies the matching metadata JSON file.

For 10X input, provide either:

`--in-dge-dir` specifies a directory containing `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.mtx.gz`.

or:

`--in-barcodes`, `--in-features`, and `--in-matrix` specify the three 10X files directly.

Multiple 10X datasets can be provided by repeating these options. In that case, `feature-vst` uses the shared feature space across datasets. `--dataset-id` can be used to label datasets.

### Unit And Feature Filtering

`--min-count-train` filters units by total raw count before computing VST statistics.

`--features` provides an optional feature list. The first column is interpreted as the feature name. If a second column is present, it may contain total counts, but `feature-vst` computes observed totals from the input data for its own HVF eligibility checks.

`--include-feature-regex` keeps only features whose names match the regex.

`--exclude-feature-regex` removes features whose names match the regex.

`--min-count-per-feature` sets the minimum observed total count required for a feature to be eligible for HVF selection.

Feature filters in `feature-vst` affect HVF eligibility, while the full `.feature.stats.tsv` file still contains all input features.

### VST Options

`--loess-span` controls the LOESS span for the mean-variance trend fit. The default is `0.3`.

`--clip-max` caps absolute standardized values before computing standardized variance. The default `-1` uses `sqrt(N)`, where `N` is the number of units.

`--n-top` controls how many HVFs are written to `.hvf.tsv`. The default `0` writes all eligible features.

`--streaming` uses a two-pass streaming implementation instead of loading all units into memory. This is useful for large inputs. For 10X input, the matrix entries must be sorted by barcode; if unsorted entries are detected, `feature-vst` exits with an error.

### IDF And Information Weights

The full statistics output includes two feature weight columns that can be used as input weights for later model fitting:

`idf` is a smoothed, normalized, capped IDF-style weight based on the raw score `log(N / (1 + n_units_present))`, where `N` is the number of units.

`info_weight` is a capped, token-mean-normalized information-content weight.

The following options control these columns:

`--idf-q` sets the percentile of raw IDF scores used for normalization. The default is `95`.

`--idf-power` sets the power used for capped IDF weighting. The default is `0.3`.

`--idf-min` sets the minimum IDF weight. The default is `0.1`.

`--idf-max` sets the cap for `info_weight`. The default is `5`.

### Output Files

`feature-vst` writes two tab-delimited files:

`<out-prefix>.feature.stats.tsv`

This file contains one row per input feature. Columns are:

`#Feature`, `Mean`, `Var`, `VarExpected`, `VarStd`, `TotalCount`, `n_units_present`, `n_units_count_gt1`, `n_units_count_gt2`, `n_units_count_gt_mean`, `idf`, `info_weight`

`Mean`, `Var`, and `TotalCount` are computed across all retained units. `VarExpected` and `VarStd` come from the VST mean-variance model. Features that are not eligible for HVF selection remain in this file.

`<out-prefix>.hvf.tsv`

This file contains the highly variable feature list. Columns are:

`#Feature`, `TotalCount`, `Score`

Rows are sorted by decreasing VST score. Only features passing `--features`, regex filters, and `--min-count-per-feature` eligibility are written.
