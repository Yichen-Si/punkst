# Poisson NMF with shifted log link

This page documents:

- `punkst nmf-pois-log1p`: fit a Poisson NMF model
- `punkst nmf-pois-log1p-transform`: project new data with a fitted model

The model is

\[
y_{im} \sim \mathrm{Poisson}(\lambda_{im}), \qquad
\log(1 + \lambda_{im}/c_i) = \sum_k \theta_{ik}\beta_{mk} + \sum_p X_{ip} B_{mp},
\]

where the covariate term is optional. By default the scale is per unit,

\[
c_i = n_i / L,
\]

with total count \(n_i\) and size factor \(L\). Use `--c` to set a fixed
constant \(c\) for all units instead.

## Input formats

Both commands accept either:

- custom sparse text input with `--in-data` and `--in-meta`
- 10X MEX input via one or more `--in-dge-dir`
- matching repeated 10X triplets: `--in-barcodes`, `--in-features`, `--in-matrix`

If both custom and 10X inputs are provided, 10X input is used. For multiple 10X
datasets, repeated `--dataset-id` values may be provided. If omitted, IDs default
to `1`, `2`, `3`, ... in input order.

For 10X fitting, data are loaded into memory. Multiple datasets are treated as a
joint corpus. If `--features` is not provided during fitting,
`nmf-pois-log1p` writes `{prefix}.features.tsv` with the final feature names and
total counts.

## `nmf-pois-log1p`

Fits a Poisson log1p NMF model.

### Required

`--K`
Number of factors.

`--out-prefix`
Prefix for output files.

One input source is required:

- custom format: `--in-data` and `--in-meta`
- 10X format: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

### Input and feature options

`--in-data`, `--in-meta`
Custom sparse input and metadata files.

`--in-dge-dir`
Input 10X MEX directory. May be repeated.

`--in-barcodes`, `--in-features`, `--in-matrix`
Input 10X MEX triplet files. Each option may be repeated.

`--dataset-id`
Dataset IDs for joint 10X input. Default: generated as `1`, `2`, `3`, ...

`--features`
Feature list used to define or filter the feature space.

`--min-count-per-feature`
Minimum feature total count. Default: `100`.

`--include-feature-regex`
Keep only matching features.

`--exclude-feature-regex`
Drop matching features.

`--min-count-train`
Minimum total count per unit for training. Default: `50`.

### Optimization options

`--mode`
Inner regression solver. `1` = TRON, `2` = FISTA, `3` = diagonal line search. Default: `1`.

`--size-factor`
Size factor \(L\) for per-unit scaling \(c_i=n_i/L\). Default: `10000`.

`--c`
Fixed scaling factor \(c\). If positive, overrides `--size-factor`. Default: unset (`-1`).

`--max-iter-outer`
Maximum alternating-update iterations. Default: `20`.

`--tol-outer`
Outer-loop convergence tolerance. Default: `1e-4`.

`--max-iter-inner`
Maximum inner regression iterations. Default: `50`.

`--tol-inner`
Inner-loop convergence tolerance. Default: `1e-6`.

`--exact`
Use exact zero-count likelihood terms instead of the approximation. Default: `false`.

`--minibatch-epoch`
Number of initial minibatch epochs. Default: `1`.

`--minibatch-size`
Minibatch size. Default: `1024`.

`--t0`
Minibatch decay offset. Default: automatic (`-1`).

`--kappa`
Minibatch decay exponent. Default: `0.7`.

`--seed`
Random seed. Default: random (`-1`).

`--threads`
Number of threads. Default: `1`.

### Warm start and supervision

`--in-model`
Warm-start from an existing model matrix.

`--random-init-missing`
Randomly initialize data features missing from the input model. Default: `false`.

`--icol-label`
Column index in `--in-covar` for guided labels. Default: unset (`-1`).

`--label-list`
List of unique labels to map to factors.

`--label-na`
String for missing labels. Default: empty string.

### Covariate options

`--in-covar`
Covariate table with one row per unit.

`--icol-covar`
Selected covariate columns, 0-based. If omitted and covariates are used, all columns except the first are used.

`--allow-na`
Replace non-numeric covariate values with zero. Default: `false`.

`--covar-coef-min`
Lower bound for covariate coefficients. Default: `-1e6`.

`--covar-coef-max`
Upper bound for covariate coefficients. Default: `1e6`.

### Diagnostics and inference

`--feature-residuals`
Compute and write per-feature residual diagnostics. Default: `false`.

`--fit-stats`
Write per-unit fit statistics. Currently skipped when `--fit-background` is enabled. Default: `false`.

`--write-se`
Write standard errors for model coefficients. Default: `false`.

`--detest-vs-avg`
Compute factor-vs-average DE statistics. Default: `false`.

`--se-method`
Standard error method: `1` = Fisher, `2` = robust, `3` = both. Default: `1`.

`--min-fc`
Minimum fold-change for DE output. Default: `1.5`.

`--max-p`
Maximum p-value for DE output. Default: `0.05`.

`--min-ct`
Minimum total count for a feature to be tested. Default: `100`.

`--verbose`
Verbose interval. Default: `500000`.

`--debug-N`
Use only the first N units. Default: `0` (disabled).

`--debug`
Debug verbosity. Default: `0`.

### Background options

`--fit-background`
Fit a background component. Default: `false`.

`--fix-background`
Keep the background model fixed during fitting. Default: `false`.

`--background-init`
Initial background proportion. Default: `0.1`.

### Outputs

`{prefix}.model.tsv`
Feature-by-factor matrix.

`{prefix}.theta.tsv`
Unit-by-factor matrix. Written when `--max-iter-inner > 0`.

`{prefix}.features.tsv`
Written for 10X fitting when `--features` is not supplied.

`{prefix}.covar.tsv`
Covariate coefficient matrix. Written when covariates are used.

`{prefix}.background.tsv`
Background model. Written when `--fit-background` is used.

`{prefix}.bgprob.tsv`
Per-unit posterior background probabilities. Written when `--fit-background` is used.

`{prefix}.model.se.fisher.tsv`, `{prefix}.model.se.robust.tsv`
Written when `--write-se` is requested for the corresponding `--se-method`.

`{prefix}.de.fisher.tsv`, `{prefix}.de.robust.tsv`
Written when `--detest-vs-avg` is requested for the corresponding `--se-method`.

`{prefix}.feature.residuals.tsv`
Written when `--feature-residuals` is used.

`{prefix}.unit_stats.tsv`
Written when `--fit-stats` is used and `--fit-background` is not set. Columns:

- `#Index`
- `TotalCount`
- `ll`
- `Residual`
- `VarMu`
- `entropy`
- `sh_lcr`
- `sh_q`

Here `entropy`, `sh_lcr`, and `sh_q` are computed from row-normalized
nonnegative factor loadings. The factor similarity matrix is cosine similarity
among L1-normalized model factors.

## `nmf-pois-log1p-transform`

Projects new data using a fitted NMF model and writes transformed unit loadings
plus diagnostics.

### Required

`--in-model`
Input model matrix from `nmf-pois-log1p`.

`--out-prefix`
Prefix for output files.

One input source is required:

- custom format: `--in-data` and `--in-meta`
- 10X format: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

### Input and covariate options

`--in-data`, `--in-meta`
Custom sparse input and metadata files.

`--in-dge-dir`
Input 10X MEX directory. May be repeated.

`--in-barcodes`, `--in-features`, `--in-matrix`
Input 10X MEX triplet files. Each option may be repeated.

`--dataset-id`
Dataset IDs for joint 10X input. Default: generated as `1`, `2`, `3`, ...

`--in-covar`
Covariate table for new data. For 10X input, rows must match the combined unit order after concatenating datasets.

`--in-covar-coef`
Covariate coefficient matrix from a previous fit.

`--icol-covar`
Selected covariate columns, 0-based. If omitted and covariates are used, all columns except the first are used.

`--allow-na`
Replace non-numeric covariate values with zero. Default: `false`.

`--min-count`
Minimum total count per unit. Default: `50`.

### Transform options

`--size-factor`
Size factor \(L\) for per-unit scaling \(c_i=n_i/L\). Default: `10000`.

`--c`
Fixed scaling factor \(c\). If positive, overrides `--size-factor`. Default: unset (`-1`).

`--max-iter-inner`
Maximum inner regression iterations. Default: `20`.

`--tol-inner`
Inner-loop convergence tolerance. Default: `1e-6`.

`--exact`
Use exact zero-count likelihood terms instead of the approximation. Default: `false`.

`--se-method`
Standard error method for transformed loadings: `1` = Fisher, `2` = robust. Default: `1`.

`--seed`
Random seed. Default: random (`-1`).

`--threads`
Number of threads. Default: `1`.

`--verbose`
Verbose interval. Default: `500000`.

`--debug-N`
Use only the first N units. Default: `0` (disabled).

`--debug`
Debug verbosity. Default: `0`.

### Outputs

`{prefix}.theta.tsv`
Projected unit-by-factor matrix. The row identifier is `#Index` for custom sparse input. For 10X input it is the barcode/identifier: unchanged for a single dataset, or `<dataset_id>:<barcode>` for multiple datasets.

`{prefix}.feature_residuals.tsv`
Per-feature averaged residuals on the projected data.

`{prefix}.unit_stats.tsv`
Per-unit summary statistics with columns:

- `#Index`
- `total_count`
- `residual`
- `ll`
- `var_mu`
- `entropy`
- `sh_lcr`
- `sh_q`

The entropy summaries use row-normalized transformed `theta` values as a
probability distribution over factors:

- `entropy = -\sum_k p_k \log p_k`
- `sh_lcr = -\sum_k p_k \log((Zp)_k)`
- `sh_q = 1 - p^T Z p`

where `Z` is the cosine similarity matrix among L1-normalized model factors.
