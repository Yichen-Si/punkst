# Poisson NMF with shifted log link

This document describes the current behavior of:

- `punkst nmf-pois-log1p`: fit a Poisson NMF model
- `punkst nmf-transform`: project new data with a fitted model

The model is

\[
y_{im} \sim \mathrm{Poisson}(\lambda_{im}), \qquad
\log(1 + \lambda_{im}/c_i) = \sum_k \theta_{ik}\beta_{mk} + \sum_p X_{ip} B_{mp},
\]

where the covariate term is optional and the per-unit scale is typically
\[
c_i = n_i / L
\]
with total count \(n_i\) and size factor \(L\).

## Input formats

Both commands accept the same count inputs:

- custom sparse text input with `--in-data` and `--in-meta`
- 10X MEX input via one or more `--in-dge-dir`
- or matching repeated 10X triplets `--in-barcodes`, `--in-features`, `--in-matrix`
- optional `--dataset-id` values if multiple 10X datasets are provided, one per dataset

For 10X input, the data are loaded into memory. When multiple datasets are provided, they are treated as one joint corpus and the default feature space is the intersection of all input feature lists. If `--features` is not provided during fitting, `nmf-pois-log1p` writes `{prefix}.features.tsv` with the final feature names and total counts. If `--dataset-id` is omitted and there are multiple datasets, dataset IDs default to `1`, `2`, `3`, ... in input order.

## `nmf-pois-log1p`

### Required

`--K`
Number of factors.

`--out-prefix`
Prefix for output files.

One input source is required:

- custom format: `--in-data` and `--in-meta`
- 10X format: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

### Common options

#### Feature and unit filtering

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

#### Optimization

`--mode`
Regression solver for internal Poisson updates. `1` = TRON, `2` = FISTA, `3` = diagonal line search.

`--size-factor`
Size factor constant \(L\). Default: `10000`.

`--c`
Use a fixed \(c\) instead of per-unit scaling.

`--max-iter-outer`
Maximum alternating-update iterations.

`--tol-outer`
Outer-loop tolerance.

`--max-iter-inner`
Maximum inner regression iterations.

`--tol-inner`
Inner-loop tolerance.

`--exact`
Use the exact likelihood for zeros instead of the approximation.

`--minibatch-epoch`
Number of initial minibatch epochs.

`--minibatch-size`
Minibatch size.

`--t0`, `--kappa`
Step-size schedule parameters for minibatch fitting.

`--seed`, `--threads`

#### Warm start and supervision

`--in-model`
Warm-start from an existing model matrix.

`--random-init-missing`
Randomly initialize features that are present in the data but missing from the input model.

`--icol-label`, `--label-list`, `--label-na`
Optional guided factorization using labels from `--in-covar`.

#### Covariates

`--in-covar`
Covariate table with one row per unit.

`--icol-covar`
Selected covariate columns. If omitted, all columns except the first are used.

`--allow-na`
Replace non-numeric covariate values with zero.

`--covar-coef-min`, `--covar-coef-max`
Bounds for covariate coefficients.

#### Optional diagnostics and inference

`--feature-residuals`
Write per-feature residual diagnostics.

`--fit-stats`
Write per-unit fit statistics. This is currently skipped when `--fit-background` is enabled.

`--write-se`
Write standard errors for model coefficients.

`--detest-vs-avg`
Compute factor-vs-average DE statistics.

`--se-method`
`1` = Fisher, `2` = robust, `3` = both.

`--min-fc`, `--max-p`, `--min-ct`
Filters for DE output.

#### Background option

`--fit-background`
Fit a background component.

`--fix-background`
Keep the background model fixed during fitting.

`--background-init`
Initial background proportion.

### Main outputs

`{prefix}.model.tsv`
Feature-by-factor matrix.

`{prefix}.theta.tsv`
Unit-by-factor matrix.

`{prefix}.features.tsv`
Written for 10X fitting when `--features` is not supplied.

`{prefix}.covar.tsv`
Written when covariates are used.

`{prefix}.background.tsv`
Written when `--fit-background` is used.

`{prefix}.bgprob.tsv`
Per-unit posterior background probabilities when `--fit-background` is used.

`{prefix}.model.se.fisher.tsv`, `{prefix}.model.se.robust.tsv`
Written when standard errors are requested.

`{prefix}.de.fisher.tsv`, `{prefix}.de.robust.tsv`
Written when DE testing is requested.

`{prefix}.feature.residuals.tsv`
Written when `--feature-residuals` is used.

`{prefix}.unit_stats.tsv`
Written when `--fit-stats` is used and `--fit-background` is not set. Current columns are:

- `#Index`
- `TotalCount`
- `ll`
- `Residual`
- `VarMu`
- `entropy`
- `sh_lcr`
- `sh_q`

Here `entropy`, `sh_lcr`, and `sh_q` are computed from the row-normalized nonnegative factor loadings of each unit. The factor similarity matrix is built from cosine similarity among the L1-normalized model factors.

## `nmf-transform`

`nmf-transform` projects new data using a fitted NMF model and writes transformed unit loadings plus diagnostics.

### Required

`--in-data`, `--in-meta`
Input counts in the custom sparse format.

Or 10X input via one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`.

`--in-model`
Input model matrix from `nmf-pois-log1p`.

`--out-prefix`

### Optional

`--in-covar`
Covariate table for the new data. For 10X input, rows must match the combined unit order after concatenating datasets in the same order they were provided.

`--in-covar-coef`
Covariate coefficient matrix from a previous fit.

`--icol-covar`, `--allow-na`

`--min-count`
Minimum total count per unit. Default: `50`.

`--mode`, `--c`, `--size-factor`, `--max-iter-inner`, `--tol-inner`, `--exact`, `--seed`, `--threads`

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

The entropy summaries use the row-normalized transformed `theta` values as a probability distribution over factors:

- `entropy = -\sum_k p_k \log p_k`
- `sh_lcr = -\sum_k p_k \log((Zp)_k)`
- `sh_q = 1 - p^T Z p`

where `Z` is the cosine similarity matrix among the L1-normalized model factors.
