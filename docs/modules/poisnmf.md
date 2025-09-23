# Poisson NMF with log(1+x) link

**`nmf-pois-log1p` fits a non-negative matrix factorization model with Poisson likelihood to single cell or spot level count data.** It fits the model $y_{im} \sim Pois(\lambda_{im}), \ \ g(\lambda_{im}) = \theta_i^T \beta_m$ where $g(\lambda_{im}) = \log(1+\lambda_{im}/c_i)$ with unit (cell) specific scaling $c_i$.

## Input format

The input data format is the same as for `topic-model`/`lda4hex`, (could be enerated from pixel level data by `tiles2hex`), including a data file and a corresponding metadata JSON file.

## Usage

```bash
punkst nmf-pois-log1p --in-data hex_data.txt --in-meta hex_meta.json \
--K 15 --out-prefix nmf_results --threads 8 --seed 1984
```

### Required

`--in-data` - Input data file (created by `tiles2hex`).

`--in-meta` - Metadata file created by `tiles2hex`.

`--K` - The number of factors (topics) to learn.

`--out-prefix` - Prefix for the output files.

### Optional

#### Data and Feature Filtering
(Similar to [`topic-model`/`lda4hex`](lda4hex.md))

`--features` - Path to a file where the first column contains gene names and the second column contains the total count of that gene. Used for filtering.

`--min-count-per-feature` - Minimum total count for a feature to be included. Requires `--features`. Default: 100.

`--include-feature-regex` - Regular expression to include only features matching this pattern.

`--exclude-feature-regex` - Regular expression to exclude features matching this pattern.

`--min-count-train` - Minimum total count for a unit (hexagon/cell) to be included in training. Default: 50.

#### Model and Training Parameters
`--mode` - Optimization algorithm to use in each Poisson regression subproblem. 1 for TRON (trust-region Newton-CG, default), 2 for Monotone FISTA (an accelerated gradient algorithm), 3 for a Newton's method with line search, 0 for Newton's method without line search.

`--size-factor` - A constant to scale the total counts per unit (matching $L$ in Seurat), used to calculate the per-observation scaling parameter $c_i=\frac{\sum_m y_{im}}{L}$ in $g(\lambda_{im}) = \log(1+\lambda_{im}/c_i)$. Default: 10000.

`--c` - If specified, use a constant `c` for all units instead of calculating it from `size-factor`. Default: -1 (per-unit `c`).

`--feature-residuals` - If set, output per-feature residuals for diagnosis.

`--max-iter-outer` - Maximum number of outer loop iterations (alternating between updating $\theta$ and $\beta$). Default: 50, but it often approximates convergence in $\lt 10$ terations.

`--max-iter-inner` - Maximum number of iterations for each Poisson regression optimization. Default: 20.

`--tol-outer` - Convergence tolerance for the outer loop. Default: 1e-5.

`--tol-inner` - Convergence tolerance for the inner optimization problems. Default: 1e-6.

`--exact` - Use the exact likelihood for all observations. If not set, uses a second-order approximatio for zero-count observations to speed up computation.

`--seed` - Random seed for reproducibility.

`--threads` - Number of threads to use. Default: 1.

#### Experimental: including covariates

We fit the model $g(\lambda_{im}) = \theta_i^T \beta_m + x_i^T b_m$ where $x_i$ contains covariates for unit $i$ provided in `--in-covar`.

Note: covariates and their effects are **not** constrained to be non-negative unless you specify so. We tried to make the optimization numerically stable but it may not always be interpretable if the non-negative $\theta_i^T \beta_m$ part is dominated by the covariate effects.

`--in-covar` - Path to a file containing covariates, with a header line. The number of rows and the **order** of units should exactly match that in the input count data `--in-data`. Currently only numerical covariates are supported, you might try to fit categorical covariates as one-hot encoded binary variables.

`--icol-covar` - Column indices (0-based) in the covariate file to use. If not specified, all columns except the first are used.

`--allow-na` - If set, non-numerical values in covariate columns are replaced with 0. Otherwise, all values must be numerical.

`--covar-coef-min` - Lower bound for covariate coefficients. Default: -1e6.

`--covar-coef-max` - Upper bound for covariate coefficients. Default: 1e6.

Example usage with covariates:

```bash
punkst nmf-pois-log1p --in-data hex_data.txt --in-meta hex_meta.json \
--K 15 --out-prefix nmf_results --threads 8 --seed 1984 \
--in-covar covars.tsv --icol-covar 1 3 5
```

#### Experimental: compute DE statistics from Cov($\hat \beta$)
`--detest-vs-avg` - Compute DE statistics for each factor vs the average, for each feature. Set to `1` to use Fisher's information based covariance, `2` to use the robust sandwich estimator, or `3` to compute both. Default: 0 (skip covariance computation so no DE output).

`--min-fc` - Minimum approximated fold change to report. Note: due to the nonlinear link, we don't have an exact fold change interpretation, the reported value is $(\exp(\beta_{km})-1)/(\exp(\beta^0_{m})-1)$ where $\beta^0_m$ is a weighted average of $\beta_m$ across factors and $\theta$ is scaled to have column sums equal to $N/K$. Default: 1.5.

`--max-p` - Maximum p-value to report. Default: 0.05.

`--min-ct` - Minimum total count for a feature to be included in DE testing. Default: 100.

### Output files

- `{prefix}.model.tsv`: The feature-factor matrix ($\beta$), where rows are features (genes) and columns are factors. It is scaled such that the column sums ($\sum_m \beta_{km}$ for each $k$) all equal to $M$, the number of features.

- `{prefix}.theta.tsv`: The unit-factor matrix ($\theta$), where rows are units (cells/hexagons) and columns are factors. The relative magnitudes of values within each row are meaningful, and the total magnitude of each row is correlated with the total counts of that unit.

- `{prefix}.fit_stats.tsv`: Per-unit (cell/hexagon) statistics including total counts, log-likelihood, residuals, and the approximated variance of the Poisson rate estimates.

- `{prefix}.covar.tsv`: (If covariates are used) The feature-covariate coefficient matrix containing $b_{jm}$ for each covariate $j$ and feature $m$.

- `{prefix}.de.{method}.tsv`: (If `--detest-vs-avg` is set) Differential expression statistics for each factor vs the average, for each feature. `{method}` is either `fisher`, `robust`, depending on `--se-method`.

- `{prefix}.feature.residuals.tsv`: (If `--feature-residuals` is set) Per-feature total residual (averaged over all data points).
