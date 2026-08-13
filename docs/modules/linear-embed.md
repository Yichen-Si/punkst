# Linear Embedding

`punkst linear-embed` constructs supervised views of dense Gamma-Poisson or
LDA topic proportions using one or more hard cluster partitions. By default it
projects both normalized raw factor proportions and their ILR transform.

## Usage

```bash
punkst linear-embed \
  --in-theta sample.results.tsv \
  --in-partition sample.initialization.results.tsv \
  --icol-partition 1 3 \
  --partition-labels kmeans leiden \
  --out-prefix sample.embed \
  --dim 2 \
  --whitening mixture
```

Required options are `--in-theta`, `--in-partition`, and `--out-prefix`.
The theta table must contain a dense trailing block of factor columns named
`0` through `K-1`. Sparse K/P theta output is not supported.

`--theta-icol-id` selects the theta identifier column and defaults to `0`.
In the partition TSV, `--icol-id` selects the identifier column and defaults
to `0`; `--icol-partition` accepts one or more partition columns and defaults
to `1`. Empty lines and lines beginning with `#` are ignored in the partition
file. Partition values may be arbitrary nonempty categorical strings.

With `--id-as-row-index`, partition identifiers are interpreted as zero-based
indices among retained theta data rows. Otherwise they are matched to the
selected theta identifier. Cluster moments are estimated from the matched
intersection, while coordinates are written for every theta row. A mismatch
warning reports the theta, partition, and intersection sizes. At least `K`
matched units and two represented clusters are required.

For multiple partition columns, `--partition-labels` may provide one unique
filename label per partition. Without explicit labels, the zero-based
partition column indices are used. A single partition uses `--out-prefix`
directly; multiple partitions write under `<prefix>.<label>`.

## Projection

`--projection-space both|linear|ilr` selects the coordinate spaces and defaults
to `both`, matching the Leiden projection interface. The `linear` space
normalizes every factor row to proportions and maps it into the Helmert
contrast subspace without flooring. The `ilr` space additionally applies
`--center-floor` (default `1e-12`) before the ILR transform.

In each selected space, hard-partition cluster weights, means, and MLE
covariance matrices define

\[
M_I=\sum_c \pi_c(\mu_c-\mu)(\mu_c-\mu)^T
\]

and

\[
M_{II}=\sum_c\pi_c(\Sigma_c-\bar\Sigma)
\Sigma^{-1}(\Sigma_c-\bar\Sigma).
\]

The `mean` view solves the generalized eigenproblem for `M_I`. The `full`
view uses `M_I Sigma^-1 M_I + M_II`. `--dim` and its alias `--visual-dim`
set the maximum number of axes (default `2`); each view is capped at its
positive numerical rank. The `mean` view is always computed; pass
`--visual-full` to additionally compute the `full` view.

`--whitening mixture` is the default and uses
`Sigma = barSigma + M_I`, which is available from the fitted partition
moments. `--whitening sample` uses every theta unit's empirical covariance in
the selected coordinate space, centered on the fitted partition mean. Sample
whitening adapts axes to the full cohort but requires an additional quadratic
covariance calculation.
`--covariance-floor` (default `1e-5`) regularizes the whitening covariance.

## Outputs

Each partition writes:

- `{prefix}.{space}.transform.tsv`: long-form projections, topic contrasts,
  eigenvalues, contrast scales, signs, and normalized weights for each selected
  space.
- `{prefix}.{space}.mean.axes.tsv`: wide interpretable topic-weight matrix for
  the always-computed mean view. `{prefix}.{space}.full.axes.tsv` is
  additionally written with `--visual-full`. Columns are `w1_p`, `w1_n`,
  `w2_p`, `w2_n`, and so on; positive and negative weights for each axis are
  separately normalized.
- `{prefix}.results.tsv`: combined `linear_mean_*` and/or `ilr_mean_*`
  coordinates, plus corresponding `*_full_*` coordinates with
  `--visual-full`, for every theta unit in theta input order.

For topic contrast coefficients `u_kd`, the wide weights are

\[
w^+_{kd}=\frac{\max(u_{kd},0)}{\sum_j\max(u_{jd},0)},\qquad
w^-_{kd}=\frac{\max(-u_{kd},0)}{\sum_j\max(-u_{jd},0)}.
\]
