# Probabilistic partition classifier

`punkst partition-classifier-fit` learns a calibrated probabilistic surrogate
for a fixed deterministic partition, such as a Leiden clustering, from dense
LDA or Gamma-Poisson topic compositions. The probabilities describe how well a
new composition reproduces that fixed reference partition conditional on the
fitted topic model. They do not include uncertainty from refitting the topic
model or clustering.

## Workflow

```bash
punkst leiden \
  --in-theta sample.results.tsv \
  --out-prefix sample.leiden

punkst partition-classifier-fit \
  --in-theta sample.results.tsv \
  --in-partition sample.leiden.clusters.tsv \
  --icol-partition 1 \
  --out-prefix sample.partition

punkst lda-transform \
  --in-state sample.state.tsv \
  --in-data new.units.tsv --in-meta new.meta.json \
  --classifier-model sample.partition.classifier.tsv \
  --out-prefix new
```

Use `gamma-pois-transform` with its fitted state for the corresponding
Gamma-Poisson workflow.

## Fitting and prediction

Fitting requires `--in-theta`, `--in-partition`, and `--out-prefix`. Theta
rows are L1-normalized without flooring. By default the theta identifier is
column 0, the partition identifier is column 0, and the partition value is
column 1. `--id-as-row-index` interprets partition identifiers as zero-based
theta data-row indices. `--icol-factor-start` and `--icol-factor-end` select a
custom consecutive factor range; otherwise trailing columns named `0..K-1`
are used.

The estimator is ridge multinomial logistic regression in normalized topic
Helmert coordinates with class Helmert contrasts. The serialized topic-space
coefficient matrix is centered across classes and within each class.

The default training sample is capped at 100,000 rows. Deterministic hashes
select rows within class-specific quotas, with a target minimum of 100 rows per
class. Retained rows receive inverse-sampling weights so prevalence represents
the full matched intersection. Every represented class needs two matched rows.

Five stratified folds, reduced to the smallest class count, select among ridge
values `1e-6,1e-5,...,1` by weighted out-of-fold log loss. A global softmax
temperature is fitted to all out-of-fold logits for the stored model;
calibration diagnostics use cross-fitted temperatures.

Outputs are:

- `{prefix}.classifier.tsv`: versioned model, topic/class order, centered
  coefficients, ridge, temperature, and sampling metadata.
- `{prefix}.results.tsv`: predictions for every theta row.
- `{prefix}.cv.tsv`: out-of-fold log loss, Brier score, and accuracy per ridge.
- `{prefix}.calibration.tsv`: overall, one-vs-rest classwise, and 15-bin
  reliability diagnostics.

Compact output is `C1 P1 C2 P2 C3 P3`. `--top-k` changes the number of pairs;
`--dense-probabilities` writes `P0..P(C-1)`. Ties use stored class order.

`partition-classifier-predict` requires `--in-theta`, `--in-model`, and
`--out-prefix`. Topic names and order must match the model exactly.

## Uncertainty-aware transforms

Pass `--classifier-model` to `lda-transform` or `gamma-pois-transform` to write
`{prefix}.classifications.tsv` and
`{prefix}.classification_diagnostics.tsv`. Unit metadata is followed by the predicted
class, maximum probability, entropy, propagation method, candidate count,
held-fixed candidate-tail mass, output top-k tail mass, LRVB status, and the
requested compact or dense probabilities. Fixed-point iterations and residual,
CG iterations, and applied curvature jitter follow the probability columns.

By default, units whose plug-in leading probability is at least 0.95 skip
uncertainty propagation. Other units use the smallest leading candidate set
with at least two classes and at least 0.999 probability mass. The excluded
tail remains fixed. Options are:

- `--classifier-top-k 3` and `--classifier-dense-probabilities`.
- `--classifier-ambiguity-threshold 0.95` and
  `--classifier-candidate-mass 0.999`.
- `--classifier-lrvb-all` to propagate every unit, or
  `--classifier-plugin-only` to disable propagation.
- `--classifier-fixed-point-tol 1e-7` and
  `--classifier-fixed-point-max-iter 5000` control local refinement.
- `--classifier-max-lrvb-failure-rate 0.01` for the aggregate attempted-unit
  failure threshold.

For a positive vector $u$, local refinement measures change as

$$
d(u',u) = \frac{\lVert u'-u\rVert_1}
{K + \max\{\lVert u'\rVert_1,\lVert u\rVert_1\}}.
$$

LDA applies this criterion to assigned topic counts. Gamma-Poisson uses the
maximum criterion over shape and rate. The default tolerance is `1e-7`, with
at most 5,000 iterations. LDA then uses the local Dirichlet shape.
Gamma-Poisson uses local shape/rate coordinates and, for dispersion-enabled
states, observed-feature exposure curvature. Matrix-free diagonally
preconditioned CG solves only the requested classifier contrasts. Numerical
jitter is limited to `1e-6` times the median positive diagonal.

One or two classifier contrasts always use 15-point Gauss-Hermite quadrature.
For higher dimensions, the second-order softmax delta method is used when its
correction is valid and small; otherwise positive-weight spherical-radial
cubature is used. Local
nonconvergence returns the original plug-in probabilities exactly. A failure
after successful refinement returns the refined plug-in probabilities. The
full stream is written
before the command returns failure if the configured aggregate rate is
exceeded.

The diagnostics file records attempted and failed LRVB units, their ratio,
the configured maximum failure rate, failure categories, and aggregate
fixed-point, CG, residual, and jitter summaries. Per-unit diagnostics use zero
iterations and a `nan` residual when refinement was not attempted.

`--classifier-bootstrap-draws` is reserved for the planned one-step
parametric-bootstrap audit and currently does not change propagation. Weighted
or non-integer data do not yet define a transform-time raw-count bootstrap
generator.

## LDA state

Plain LDA fitting writes a versioned `{prefix}.state.tsv` containing global
SVB components, alpha, eta, ordered topic/feature names, and feature weights.
`lda-transform --in-state` restores them. Legacy `--in-model` remains
supported; LRVB classification with it requires explicit positive `--alpha`.
Background-enabled LDA is not represented by this state and is unsupported for
LRVB classification.
