# Probabilistic partition classifier

`punkst partition-classifier-fit` fits a ridge multinomial logistic regression for predicting a given partition from factor compositions. It serves as a probabilistic surrogate for the deterministic partition, such as a Leiden clustering.

## Workflow

```bash
punkst leiden \
  --in-theta sample.results.tsv \
  --out-prefix sample.leiden --threads 4

punkst partition-classifier-fit \
  --in-theta sample.results.tsv \
  --in-partition sample.leiden.clusters.tsv \
  --icol-partition 1 \
  --out-prefix sample.partition \
  --crossfit --threads 4

punkst lda-transform \
  --in-state sample.state.tsv \
  --in-data new.units.tsv --in-meta new.meta.json \
  --classifier-model sample.partition.classifier.tsv \
  --out-prefix new --threads 4
```

Use `gamma-pois-transform` with its fitted state for the corresponding
Gamma-Poisson workflow.

## Fit a classifier

The fit command matches rows in a factor-composition table to labels in a
partition table. It needs at least two matched units in every retained class.
By default, it combines an ordinary linear classifier with a small quadratic
component. This can represent curved boundaries between partition labels while
keeping the additional model small.
It writes a reusable model at `{prefix}.classifier.tsv`, predictions for the
input rows at `{prefix}.classifications.tsv`, and validation and calibration
summaries at `{prefix}.cv.tsv` and `{prefix}.calibration.tsv`.

### Required options

- `--in-theta FILE`: dense factor-composition table from the fitted or
  transformed model.
- `--in-partition FILE`: TSV containing the reference partition labels.
- `--out-prefix PREFIX`: prefix for all classifier outputs.

### Match input rows and factors

- `--theta-icol-id N` (default: `0`): zero-based identifier column in
  `--in-theta`.
- `--icol-id N` (default: `0`): zero-based identifier column in
  `--in-partition`.
- `--icol-partition N` (default: `1`): zero-based label column in
  `--in-partition`.
- `--id-as-row-index`: treat partition identifiers as zero-based row indices in
  the theta table instead of matching identifier strings.
- `--icol-factor-start N` and `--icol-factor-end N`: select the inclusive,
  zero-based range of factor columns. Supply both options together. By default,
  the command uses the table's trailing numeric factor columns.
- `--factor-weight-threshold X` (default: `1e-5`): keep factors whose average
  weight across all L1-normalized theta rows is strictly greater than `X`.
  Filtering happens before row matching and applies to both parts of the
  classifier. Set `X` to zero or a negative value to keep every factor. At
  least two factors must remain.

### Control fitting

- `--train-max-rows N` (default: `100000`): maximum number of matched rows
  used to fit the classifier. The sample is chosen reproducibly across classes.
- `--min-per-class N` (default: `100`): target minimum number of sampled rows
  per class; must be at least `2`.
- `--folds N` (default: `5`): maximum number of cross-validation folds; must
  be at least `2`.
- `--ridge-grid V1 V2 ...`: ridge values to consider, supplied as one or more
  space-separated values. If omitted, the built-in grid is used.
- `--quadratic-rank N` (default: `8`): size of the reduced factor space used
  by the quadratic component. It is capped by the numbers of retained factors
  and classes. The linear component continues to use every retained factor.
  Set `N=0` to fit the original linear-only classifier.
- `--max-iterations N` (default: `300`): maximum optimizer iterations per fit.
- `--lbfgs-history N` (default: `10`): optimizer history size.
- `--gradient-tolerance X` (default: `1e-7`): optimizer convergence tolerance.
- `--threads N` (default: `1`): number of parallel classifier fits. Use `0` for
  the runtime default.
- `--sampling-seed N` (default: `1`): non-negative seed for reproducible
  sampling and fold assignment.
- `--crossfit`: also create a crossfit bundle. This is useful when predictions
  may include units used to train the classifier, but takes longer to fit. Each
  validation model learns its reduced quadratic space only from its training
  rows.

### Choose prediction output

- `--top-k N` (default: `3`): write the top `N` class/probability pairs for
  each unit.
- `--dense-probabilities`: write a probability column for every class instead
  of top class/probability pairs.

With `--crossfit`, the command also writes
`{prefix}.crossfit.classifier.tsv`, crossfit predictions and disagreements,
and `{prefix}.crossfit.diagnostics.tsv`. The ordinary
`{prefix}.classifier.tsv` is always produced and is normally the right model
for a new dataset.

## Incorporate classification during a factor transform

For `lda-transform` or `gamma-pois-transform`, pass
`--classifier-model sample.partition.classifier.tsv` to write classifications
alongside the transform results. Their full option references are documented in
[lda4hex](lda4hex.md) and [gamma-pois](gammapois.md).

To use a crossfit bundle on data that overlaps the fitting cohort, pass the
bundle as `--classifier-model` and add `--classifier-crossfit`. For a new,
unrelated dataset, use the ordinary model or the bundle without that flag.

The classification takes into account the uncertainty in factor composition, and is more justified than the point-estimate based version below.

## Predict from a saved classifier

Use `partition-classifier-predict` to apply a trained classifier to a factor-composition table treating the latter as fixed observations.

```bash
punkst partition-classifier-predict \
  --in-theta new.results.tsv \
  --in-model sample.partition.classifier.tsv \
  --out-prefix new.partition
```

The factor names and their order in `--in-theta` must match the saved model.
The command writes `{prefix}.results.tsv`.

Options:

- `--in-theta FILE` (required): dense factor-composition table to classify.
- `--in-model FILE` (required): classifier model created by
  `partition-classifier-fit`.
- `--out-prefix PREFIX` (required): prefix for the prediction output.
- `--theta-icol-id N` (default: `0`): zero-based identifier column in
  `--in-theta`.
- `--top-k N` (default: `3`): write the top `N` class/probability pairs.
- `--dense-probabilities`: write a probability column for every class instead.
- `--threads N` (default: `1`): number of prediction workers. Use `0` for the
  runtime default.

With more than one threads, output rows may not follow input order. Use the identifier column when joining predictions to other tables.
