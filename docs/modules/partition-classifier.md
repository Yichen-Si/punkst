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
  --out-prefix sample.partition \
  --crossfit

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

`--crossfit` additionally fits strict nested outer-fold models. For each outer
fold, ridge selection and temperature calibration use only the outer training
rows; the outer held-out rows do not enter coefficients, hyperparameter
selection, or calibration. Fold assignment is deterministic, stratified by
class, and keyed by unit identifier. This evaluates the supervised classifier
conditional on the already fitted topic model and fixed partition; it does not
cross-fit either of those upstream stages.

Crossfit fitting is substantially more expensive than the ordinary fit because
each outer model performs its own inner cross-validation. The ordinary
all-data model is still fitted and stored in the bundle, so one bundle supports
both prediction of genuinely new units and honest classifier-stage predictions
for sampled fitting-cohort units.

Outputs are:

- `{prefix}.classifier.tsv`: versioned model, topic/class order, centered
  coefficients, ridge, temperature, and sampling metadata.
- `{prefix}.classifications.tsv`: plug-in full-model classifications for every
  theta row, written as `id C1 P1 C2 P2 ...` (or dense probabilities).
- `{prefix}.cv.tsv`: out-of-fold log loss, Brier score, and accuracy per ridge.
- `{prefix}.calibration.tsv`: overall, one-vs-rest classwise, and 15-bin
  reliability diagnostics.
- `{prefix}.crossfit.classifier.tsv`: self-contained bundle containing the
  ordinary full model, all outer-fold models, and sampled-identifier routes
  (with `--crossfit`).
- `{prefix}.crossfit.classifications.tsv`: out-of-sample plug-in
  classifications for every theta row. Sampled training identifiers use the
  outer-fold model that excluded them; other rows use the full model, which
  was fitted only on the sampled training identifiers.
- `{prefix}.crossfit.diagnostics.tsv`: per-outer-fold training/held-out counts,
  selected ridge, calibrated temperature, and unpenalized held-out metrics,
  plus their aggregate.

Compact output is `C1 P1 C2 P2 C3 P3`. `--top-k` changes the number of pairs;
`--dense-probabilities` writes `P0..P(C-1)`. Ties use stored class order.

`partition-classifier-predict` requires `--in-theta`, `--in-model`, and
`--out-prefix`. Topic names and order must match the model exactly.

## Uncertainty-aware transforms

Pass `--classifier-model` to `lda-transform` or `gamma-pois-transform` to write
`{prefix}.classifications.tsv` and
`{prefix}.classification_diagnostics.tsv`. Unit metadata is followed by
entropy, propagation method, candidate count, held-fixed candidate-tail mass,
output top-k tail mass, LRVB status, and the requested compact or dense
probabilities. In compact output, `C1` and `P1` are the predicted class and its
maximum probability. Fixed-point iterations and residual, CG iterations, and
applied curvature jitter follow the probability columns.

By default, `{prefix}` is the value of `--out-prefix`. Set
`--out-prefix-classifier` to place the two classification files under a
different prefix while leaving topic-transform results, pseudobulk, and
residual outputs under `--out-prefix`.

To apply multiple classifiers to one factor transform, add
`--classifier-only --in-transform-results prior.results.tsv`. The command
rereads the original counts and fitted factor state, takes dense topic
compositions from the prior result in unit order, and writes only classifier
outputs. Confident units stop after plug-in prediction; ambiguous units use the
saved composition as a warm start and continue local inference to the stricter
classifier fixed point before LRVB. The source table must have the expected
topics as its exact trailing dense block, and the input filtering, weights,
minimum count, modality, and identifier settings must reproduce the source
transform. This is a convergent warm start, not an exact posterior checkpoint.

Gamma-Poisson classifier-only runs must also select the original dispersion:
use `--factor-is-in-sample` or `--use-stored-dispersion` when it came from the
fitted state, or pass the original transform diagnostics with
`--in-transform-dispersion prior.dispersion.tsv`.

`--classifier-model` accepts either the legacy single-model file or a crossfit
bundle. A bundle uses its embedded all-data model by default, which is the
appropriate choice for a new dataset. Add `--classifier-crossfit` explicitly
when transforming data that may overlap the classifier fitting cohort. Sampled
fitting identifiers then use their held-out outer-fold models; identifiers not
stored in the bundle use the all-data model. This explicit opt-in avoids
accidentally treating coincident identifiers from an unrelated dataset as
training rows. `classifier_model_source` is `full`, `heldout_fold_N`, or
`full_unseen` on every classification row. The diagnostics file also reports
the prediction mode and routed row counts, and a crossfit run warns when no
identifier matched.

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
