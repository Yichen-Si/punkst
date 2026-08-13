# Linear Embedding

`punkst linear-embed` constructs supervised linear embedding of LDA or Gamma-Poisson topic proportions given one or more hard cluster partitions.

## Usage

```bash
punkst linear-embed --dim 4 \
  --in-theta sample.results.tsv \
  --in-partition sample.labels.tsv \
  --icol-partition 1 3 \
  --partition-labels label1 label2 \
  --out-prefix sample.embed
```

**Required:**

`--in-theta`, `--in-partition`, and `--out-prefix`.

The "theta" table must contain consecutive factor columns named `0` through
`K-1`. Alternatively, `--icol-factor-start` and `--icol-factor-end` select an
inclusive zero-based range of at least two theta columns. Both options must be
supplied together, and the selected header names are retained as factor names.

`--theta-icol-id` selects the theta identifier column index and defaults to `0`.

In the TSV file that contains partitions, `--icol-id` selects the identifier column and defaults to `0`; `--icol-partition` accepts one or more partition columns and defaults to `1`. Partition values may be arbitrary nonempty strings (as cluster labels).

With `--id-as-row-index`, partition identifiers are interpreted as zero-based
indices among retained theta data rows. Otherwise they are matched to the
identifiers in the theta file. Projection is learned from the matched
intersection, while coordinates are written for every theta row.

For multiple partition columns, `--partition-labels` may provide one unique
label per partition to identify output files. Without explicit labels, the zero-based
partition column indices are used. A single partition writes output files with prefix specified by `--out-prefix`; multiple partitions write `<prefix>.<label>.*`.

## Projection

`--projection-space linear|both` selects the coordinate spaces and defaults to
`linear`. The linear space normalizes every factor row to proportions and maps
it into the Helmert contrast subspace without flooring. Pass
`--projection-space both` to additionally compute mean/full projections in ILR
space; this applies `--center-floor` (default `1e-12`) before the ILR
transform. ILR-only output is not supported.

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

`--whitening mixture` is the default and uses `Sigma = barSigma + M_I`, which is available from the fitted partition moments. `--whitening sample` uses every theta unit's empirical covariance thus requiring an additional quadratic
covariance calculation. `--covariance-floor` (default `1e-5`) regularizes the whitening covariance.

### Conditional-QDA projection

By default, the command additionally learns a lower-dimensional orthonormal
projection of `theta` that optimizes conditional quadratic discriminant
analysis (QDA) log loss on the training partition. It is only fit in the
linear space. Pass `--skip-qda-projection` to disable it.

QDA dimensions are capped at `K-2`, and every class needs at least three
matched rows. Default parameters are:

- `--qda-train-max-rows 12000` is a hard training-row ceiling; `0` disables
  only this hard ceiling. The actual training cap is also bounded by the
  adaptive rule below. `--qda-validation-max-rows 4000`; `0` means no
  validation cap.
- `--qda-validation-fraction 0.2`, applied within each class before caps.
- `--qda-epochs 250`, `--qda-learning-rate 0.03`, and `--qda-restarts 2`.
- `--qda-covariance-shrinkage 0.10` and `--qda-ridge 1e-5`.
- `--qda-eval-every 5`, `--qda-patience 12`, and `--qda-seed 1`.

The final basis is rotated within its learned subspace by descending projected
between-class scatter and assigned deterministic signs. This preserves QDA
probabilities while giving the otherwise non-identifiable axes a stable order.

<!-- For `p = K-1` input dimensions, `d` output dimensions, and `C` classes, the
adaptive training cap is

\[
\max\left(50C,\;3\left[d(p-d)+C\frac{d(d+3)}{2}+(C-1)\right]\right).
\]

The number of training rows used is the minimum of the available training
rows, this adaptive cap, and the nonzero hard ceiling. Diagnostics report all
three cap values. -->

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

Unless `--skip-qda-projection` is supplied, the results table also contains
`linear_qda_*` coordinates and the command writes:

- `{prefix}.linear.qda.transform.tsv`: Helmert-coordinate coefficients and
  corresponding topic contrasts.
- `{prefix}.linear.qda.axes.tsv`: positive/negative normalized topic weights.
- `{prefix}.linear.qda.diagnostics.tsv`: selected restart and epoch, losses,
  row counts, seed, and optimizer settings.

For topic contrast coefficients `u_kd`, the wide weights are

\[
w^+_{kd}=\frac{\max(u_{kd},0)}{\sum_j\max(u_{jd},0)},\qquad
w^-_{kd}=\frac{\max(-u_{kd},0)}{\sum_j\max(-u_{jd},0)}.
\]

## Optional PyTorch reference

`ext/py/qda_projection.py` implements the same QDA projection as a standalone CPU/CUDA/MPS reference. It is useful for GPU experiments; the punkst binary does not call it and has no PyTorch dependency.

```bash
python -m pip install numpy torch
python ext/py/qda_projection.py \
  --in-theta sample.results.tsv \
  --in-partition sample.initialization.results.tsv \
  --icol-partition 1 \
  --out-prefix sample.qda \
  --dim 2 --device auto
```

The Python utility mirrors the native defaults, split/cap policy, and QDA
sidecar schemas.
