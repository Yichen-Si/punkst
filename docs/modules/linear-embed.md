# Linear Embedding

`punkst linear-embed` constructs supervised linear embedding of LDA or Gamma-Poisson topic proportions given one or more hard cluster partitions. The goal is to lean interpretable low-dimensional visualizations where each axis is a contrast of topics. It may not separate clusters as well as nonlinear methods, and 2 dimensions may be insufficient for complex datasets.

## Usage

```bash
punkst linear-embed --dim 4 \
  --in-theta sample.results.tsv \
  --in-partition sample.labels.tsv \
  --icol-partition 1 3 \
  --partition-labels label1 label2 \
  --out-prefix sample.embed
```

`--dim` and its alias `--visual-dim` set the embedding dimension (default `4`).

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

## Projection options

`linear-embed` and Leiden's post-clustering embeddings use the same projection
pipeline and defaults. Option names are shared unless a Leiden spelling is
shown below.

`--projection-space linear|both`
: Select the coordinate spaces. `linear` (the default) writes direct
  simplex-space projections. `both` additionally writes the ILR-space mean
  view and, when requested, the ILR full view. ILR-only output is not
  supported. The QDA projection is always fitted in linear contrast space and
  is controlled separately by `--skip-qda-projection`.

`--dim D`, `--visual-dim D` (Leiden: `--projection-dim D`)
: Set the requested number of axes (default `4`). `--visual-dim` is a
  backward-compatible `linear-embed` alias. The actual eigendecomposition
  dimension is capped by `K-1`, one less than the number of represented
  clusters, and the positive numerical rank of the between-cluster kernel;
  QDA is capped at `K-2`.

`--whitening mixture|sample` (Leiden: `--projection-whitening mixture|sample`)
: Select the whitening covariance for eigendecomposition projections.
  `mixture` (the default) uses the total mixture covariance available from the
  fitted partition moments. `sample` uses every theta row's empirical
  covariance and therefore requires an additional quadratic covariance
  calculation.

`--center-floor X` (Leiden: `--projection-center-floor X`)
: Set the positive floor applied before the ILR transform (default `1e-12`).
  It does not affect linear projections.

`--covariance-floor X` (Leiden: `--projection-covariance-floor X`)
: Set the positive eigenvalue floor used to regularize the whitening
  covariance (default `1e-5`).

`--visual-full` (Leiden: `--projection-full`)
: Also compute the full covariance-shape eigendecomposition view.

`--skip-qda-projection`
: Disable the conditional-QDA projection, which is computed by default. The
  QDA optimization, sampling, sparsity, and cross-validation controls are
  described under [Conditional-QDA projection](#conditional-qda-projection).

Leiden additionally accepts `--skip-projection` to skip all projection work.
Projection never changes its graph or community assignments. With Leiden's
`--allow-topk` input, omitted topics are zero-filled before projection, so
coordinates describe the reconstructed top-k composition rather than the
original full vector.

## Projection methods

### Maximize between-class scatter

Define the kernel for the between-class mean difference
\[
M_I=\sum_c \pi_c(\mu_c-\mu)(\mu_c-\mu)^T
\]

and the kernel for the between-class covariance difference

\[
M_{II}=\sum_c\pi_c(\Sigma_c-\bar\Sigma)
\Sigma^{-1}(\Sigma_c-\bar\Sigma).
\]

The `mean` view solves the generalized eigenproblem for `M_I`. The `full`
view uses `M_I Sigma^-1 M_I + M_II`. The `mean` view is always computed, while
the `full` view is optional. (We observed that $M_I$ often works better, though
$M_{II}$ and $M_I+M_{II}$ are proposed in the literature.)

Mixture whitening uses `Sigma = barSigma + M_I`; sample whitening instead uses
the empirical covariance over theta rows. The configured covariance floor
regularizes either whitening covariance.

The linear space normalizes every factor row to proportions and maps it into
the Helmert contrast subspace. The optional ILR views apply the configured
center floor before the ILR transform.

### Conditional-QDA projection

The command also learns a lower-dimensional orthonormal projection of `theta`
that optimizes conditional quadratic discriminant analysis (QDA) log loss in
the lower-dimensional space.

QDA dimensions are capped at `K-2`. Before fitting, clusters with 10 or fewer
rows in the matched theta/partition intersection are discarded from the QDA
training population. The command reports retained and represented cluster
counts before optimization. If fewer than two eligible clusters remain, QDA
is omitted with a warning while mean/full projections continue to use every
represented cluster. Default parameters are:

- `--qda-train-max-rows 12000` is a hard training-row ceiling; `0` disables
  only this hard ceiling. The actual training cap is also bounded by an
  adaptive rule depending on $K$, $C$, and the embedding dimension.
- `--qda-validation-max-rows 4000`; `0` means no validation cap.
- `--qda-validation-fraction 0.2`, applied within each class before caps.
- `--qda-epochs 250`, `--qda-learning-rate 0.03`, and `--qda-restarts 2`.
- `--qda-covariance-shrinkage 0.10` and `--qda-ridge 1e-5`.
- `--qda-eval-every 5`, `--qda-patience 12`, and `--qda-seed 1`.

`--qda-sparsity-strength` adds an optional quartimax reward to the training
objective. For projection `Q`, normalized Helmert matrix `H`, and output
dimension `d`, the optimized objective is

\[
\mathcal L_{\mathrm{QDA}}(Q)
-\frac{\lambda}{d}\sum_{k,j}(H^\top Q)_{kj}^4.
\]

The default `lambda=0` leaves subspace learning unregularized. A positive value
can trade predictive log loss for a subspace with more topic-concentrated
contrasts.

Pass `--qda-sparsity-cv` to select `lambda` by nested stratified
cross-validation. The default grid is
`0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1`; replace it with
`--qda-sparsity-grid`. `--qda-sparsity-cv-folds` defaults to five. Each outer
fold is scored using unpenalized conditional log loss and is not used for
optimizer stopping. Inner validation selects optimizer checkpoints. The
largest `lambda` whose mean outer-fold loss is no greater than the `lambda=0`
mean plus its fold-level standard error is selected, then refit on the normal
training/validation split. CV costs approximately the number of folds times
the number of grid values in QDA fits.

The final basis is rotated within its learned subspace using a deterministic
quartimax rotation of the topic contrasts. This maximizes the sum of fourth
powers of the contrast coefficients, favoring axes dominated by fewer topics
without thresholding coefficients. Axes are then ordered by descending
projected between-class scatter and signed so that the largest absolute topic
contrast is positive. The rotation preserves the learned subspace, pairwise
distances, and QDA probabilities and log loss.

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

- `{prefix}.cluster_labels.tsv`: the internal zero-based cluster index and its
  original string value from the partition column. Indices follow first
  appearance in the matched theta/partition intersection and are the labels
  used by the embedding implementation.
- `{prefix}.cluster_factors.tsv`: factors as rows and internal zero-based
  clusters as columns. Values are unnormalized sums of the input theta values
  over matched units assigned to each cluster. Unmatched theta and partition
  rows do not contribute because they have no paired cluster assignment.
- `{prefix}.{space}.transform.tsv`: long-form projections, topic contrasts,
  rotated-axis separation scores, contrast scales, signs, and normalized
  weights for each selected space. After the top eigenspace is selected, its
  axes are always orthogonally quartimax-rotated to make the topic contrasts
  sparse. This preserves the embedding subspace and pairwise distances; a
  separation score is the rotated axis's Rayleigh score, not an eigenvalue.
- `{prefix}.{space}.mean.axes.tsv`: wide interpretable topic-weight matrix for
  the always-computed mean view. `{prefix}.{space}.full.axes.tsv` is
  additionally written with `--visual-full`. Columns are `w1_p`, `w1_n`,
  `w2_p`, `w2_n`, and so on; positive and negative weights for each axis are
  separately normalized. The row-label header is `#Cluster`.
- `{prefix}.results.tsv`: combined `linear_mean_*` and/or `ilr_mean_*`
  coordinates, plus corresponding `*_full_*` coordinates with
  `--visual-full`, for every theta unit in theta input order.

Unless `--skip-qda-projection` is supplied, the results table also contains
`linear_qda_*` coordinates and the command writes:

- `{prefix}.linear.qda.transform.tsv`: Helmert-coordinate coefficients and
  corresponding topic contrasts.
- `{prefix}.linear.qda.axes.tsv`: positive/negative normalized topic weights.
- `{prefix}.linear.qda.diagnostics.tsv`: selected restart and epoch, losses,
  row counts, sparsity score and strength, seed, and optimizer settings.
- `{prefix}.linear.qda.sparsity_cv.tsv`: when `--qda-sparsity-cv` is used,
  the tested strengths, raw held-out losses and standard errors, quartimax
  scores, eligibility threshold, and selected strength.

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
