# Poisson NMF with log(1+x) link

This document describes two related commands: `nmf-pois-log1p` for fitting a Poisson NMF model, and `nmf-transform` for applying a fitted model to new data.

## `nmf-pois-log1p`: Fitting the model

**`nmf-pois-log1p` fits a non-negative matrix factorization model with Poisson likelihood to single cell or spot level count data.** It fits the model $y_{im} \sim Pois(\lambda_{im}), \ \ g(\lambda_{im}) = \theta_i^T \beta_m$ where $g(\lambda_{im}) = \log(1+\lambda_{im}/c_i)$ with unit (cell) specific scaling $c_i := n_i/L$. ($n_i$: total transcript count of unit $i$. $L$: constnat size factor)

### Input format

The input data format is the same as for `topic-model` (could be enerated from pixel level data by `tiles2hex`), including a data file and a corresponding metadata JSON file.

### Usage

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
(Similar to [`topic-model`](lda4hex.md))

`--features` - Path to a file where the first column contains gene names and the second column contains the total count of that gene. Used for filtering.

`--min-count-per-feature` - Minimum total count for a feature to be included. Requires `--features`. Default: 100.

`--include-feature-regex` - Regular expression to include only features matching this pattern.

`--exclude-feature-regex` - Regular expression to exclude features matching this pattern.

`--min-count-train` - Minimum total count for a unit (hexagon/cell) to be included in training. Default: 50.

#### Input for warm-start or guided mode

`--in-model` - Input model (beta) file for warm start.

`--random-init-missing` - If set, randomly initialize features present in the data but missing from the input model.

`--icol-label` - Column index in `--in-covar` for labels to guide the factorization. This turns the model into a guided/supervised one where factors correspond to labels.

`--label-list` - A file containing a list of unique labels, one per line. The order determines the factor indices.

`--label-na` - String for missing labels in the label column (e.g., "NA"). These units are not used for guidance.

#### Model and Training Parameters
`--mode` - Optimization algorithm to use in each Poisson regression subproblem. 1 for TRON (trust-region Newton-CG, default), 2 for an accelerated gradient algorithm (FISTA), 3 for a Newton's method with line search.

`--size-factor` - A constant to scale the total counts per unit (matching $L$ in Seurat), used to calculate the per-observation scaling parameter $c_i=\frac{\sum_m y_{im}}{L}$ in $g(\lambda_{im}) = \log(1+\lambda_{im}/c_i)$. Default: 10000.

`--c` - If specified, use a constant `c` for all units instead of calculating it from `size-factor`. Default: -1 (per-unit `c`).

`--feature-residuals` - If set, output per-feature residuals for diagnosis.

`--fit-stats` - If set, compute and write goodness of fit statistics. Not compatible with `--fit-background`.

`--write-se` - If set, compute and write standard errors for the model parameters ($\beta$). See also `--se-method`.

`--max-iter-outer` - Maximum number of outer loop iterations (alternating between updating $\theta$ and $\beta$). Default: 50, but it often approximates convergence in $\lt 10$ iterations.

`--max-iter-inner` - Maximum number of iterations for each Poisson regression optimization. Default: 20.

`--tol-outer` - Convergence tolerance for the outer loop. Default: 1e-5.

`--tol-inner` - Convergence tolerance for the inner optimization problems. Default: 1e-6.

`--exact` - Use the exact likelihood for all observations. If not set, uses a second-order approximation for zero-count observations to speed up computation.

`--seed` - Random seed for reproducibility.

`--threads` - Number of threads to use. Default: 1.

#### Minibatch Training Options

`--minibatch-epoch` - Number of minibatch epochs at the beginning of training. Default: 1.

`--minibatch-size` - Minibatch size. Default: not set (full batch).

`--t0` - Decay parameter t0 for minibatch step size. Default: 1.

`--kappa` - Decay parameter kappa for minibatch step size. Default: 0.9.

#### Experimental: Background Noise Modeling

`--fit-background` - If set, fit a background noise component.

`--fix-background` - If set, fix the background model during training (useful for warm starts).

`--background-init` - Initial background proportion (pi0). Default: 0.1.

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
`--detest-vs-avg` - If set, compute DE statistics for each factor vs the average, for each feature.

`--se-method` - Method for calculating standard errors of $\beta$: `1` for Fisher's information based covariance (default), `2` for the robust sandwich estimator, or `3` to compute both. Used for DE testing or if `--write-se` is set.

`--min-fc` - Minimum approximated fold change to report. Note: due to the nonlinear link, we don't have an exact fold change interpretation, the reported value is $(\exp(\beta_{km})-1)/(\exp(\beta^0_{m})-1)$ where $\beta^0_m$ is a weighted average of $\beta_m$ across factors and $\theta$ is scaled to have column sums equal to $N/K$. Default: 1.5.

`--max-p` - Maximum p-value to report. Default: 0.05.

`--min-ct` - Minimum total count for a feature to be included in DE testing. Default: 100.

### Output files

- `{prefix}.model.tsv`: The feature-factor matrix ($\beta$), where rows are features (genes) and columns are factors. It is scaled such that the column sums ($\sum_m \beta_{km}$ for each $k$) all equal to $M$, the number of features.

- `{prefix}.model.se.{method}.tsv`: (If `--write-se` is set) Standard errors of $\beta$. `{method}` is either `fisher` or `robust`.

- `{prefix}.theta.tsv`: The unit-factor matrix ($\theta$), where rows are units (cells/hexagons) and columns are factors. The relative magnitudes of values within each row are meaningful, and the total magnitude of each row is correlated with the total counts of that unit.

- `{prefix}.fit_stats.tsv`: (If `--fit-stats` is set) Per-unit (cell/hexagon) statistics including total counts, log-likelihood, residuals, and the approximated variance of the Poisson rate estimates.

- `{prefix}.covar.tsv`: (If covariates are used) The feature-covariate coefficient matrix containing $b_{jm}$ for each covariate $j$ and feature $m$.

- `{prefix}.de.{method}.tsv`: (If `--detest-vs-avg` is set) Differential expression statistics for each factor vs the average, for each feature. `{method}` is either `fisher` or `robust`.

- `{prefix}.feature.residuals.tsv`: (If `--feature-residuals` is set) Per-feature total residual (averaged over all data points).

- `{prefix}.background.tsv`: (If `--fit-background` is set) Background expression rates for each feature.

- `{prefix}.bgprob.tsv`: (If `--fit-background` is set) Per-unit posterior probabilities of belonging to the background component.


## `nmf-transform`: Projecting new data with a trained model

The `nmf-transform` command applies a trained NMF model to new data to obtain the unit-factor matrix ($\theta$) for the new units.

### Usage

```bash
punkst nmf-transform --in-data new_data.txt --in-meta new_meta.json \
--in-model nmf_results.model.tsv --out-prefix projected_results
```

### Required

`--in-data` - Input data file for the new data.

`--in-meta` - Metadata file for the new data.

`--in-model` - The trained feature-factor matrix (`.model.tsv` file from `nmf-pois-log1p`).

`--out-prefix` - Prefix for the output files.

### Optional

#### Input Data and Model

`--in-covar` - Covariate file for the new data.

`--in-covar-coef` - The trained feature-covariate coefficient matrix (`.covar.tsv` file from `nmf-pois-log1p`).

`--allow-na` - If set, replace non-numerical values in covariates with zero.

`--icol-covar` - Column indices (0-based) in `--in-covar` to use.

`--min-count` - Minimum total count for a unit to be included. Default: 50.

#### Algorithm Parameters

`--mode` - Optimization algorithm. 1 for TRON (default), 2 for FISTA, 3 for a Newton method with line search.

`--c` - Constant `c` in `log(1+lambda/c)`. If not set, it is calculated from `size-factor`.

`--size-factor` - Constant to scale total counts to get per-unit scaling factors. Default: 10000.

`--max-iter-inner` - Maximum number of iterations. Default: 20.

`--tol-inner` - Convergence tolerance. Default: 1e-6.

`--exact` - Use exact likelihood for all observations.

`--seed` - Random seed.

`--threads` - Number of threads. Default: 1.

### Output files

- `{prefix}.theta.tsv`: The transformed unit-factor matrix ($\theta$) for the new data.

- `{prefix}.fit_stats.tsv`: Per-unit goodness-of-fit statistics for the new data.

- `{prefix}.feature_residuals.tsv`: Per-feature averaged residuals on the new data.
