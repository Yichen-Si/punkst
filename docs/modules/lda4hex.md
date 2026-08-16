# Latent Dirichlet Allocation (LDA)

This document describes the current behavior of:

- `punkst topic-model`
- `punkst lda-transform`

Dense `{prefix}.results.tsv` output can be clustered directly with
[`punkst leiden`](leiden.md).

## Example usage

Fit a 24-topic model from custom sparse input and transform the same units:

```bash
punkst topic-model \
  --in-data sample.units.tsv --in-meta sample.meta.json \
  --out-prefix sample.lda --residuals --transform \
  --n-topics 24 --n-epochs 3 \
  --threads 4 --seed 1
```

Apply the fitted model to another dataset:

```bash
punkst lda-transform \
  --in-data new.units.tsv --in-meta new.meta.json \
  --in-model sample.lda.model.tsv \
  --out-prefix new.lda --residuals \
  --threads 4 --seed 1
```

For 10X MEX input, replace `--in-data` and `--in-meta` with one or more
`--in-dge-dir` values, or provide matching `--in-barcodes`, `--in-features`,
and `--in-matrix` lists.

## Input formats

### Custom sparse text input

This is the sparse unit-by-feature format produced by `tiles2hex`.

Each line contains the sparse encoding for one unit. The metadata JSON supplies:

- `dictionary`: feature name to feature index mapping, unless you instead provide the feature list explicitly with `--features`
- `offset_data`: number of prefix columns before the sparse payload
- `header_info`: names of those prefix columns

The sparse payload contains:

- number of nonzero features
- total count for the unit
- `(feature_index, count)` pairs

### 10X MEX input

You can also use:

- one or more `--in-dge-dir` values (For example, `--in-dge-dir path/d1 path/d2` where each directory contains `matrix.mtx.gz`, `features.tsv.gz`, and `barcodes.tsv.gz`)
- or matching lists for `--in-barcodes`, `--in-features`, and `--in-matrix` (For example, `--in-barcodes path/d1/barcodes.tsv.gz path/d2/barcodes.tsv.gz --in-features path/d1/features.tsv.gz path/d1/features.tsv.gz --in-matrix path/d1/matrix.mtx.gz path/d2/matrix.mtx.gz`)
- optional `--dataset-id` values, one per dataset (For example, `--dataset-id d1 d2` in the above example)

For 10X input, the data are loaded into memory. When multiple datasets are provided, they are treated as one joint corpus and the default feature space is the intersection of all input feature lists. When `--features` is not supplied during fitting, `{prefix}.features.tsv` is written with the final feature names and total counts.

If `--dataset-id` is omitted, dataset IDs default to `1`, `2`, `3`, ... in input order.

## `topic-model`

### Required

- Output prefix `--out-prefix`

Either:

- custom input: `--in-data` and `--in-meta`
- 10X input: one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

And one of:

- `--n-topics`
- `--model-prior`

If `--projection-only` is used, `--model-prior` is required and no training is performed.

### Feature selection and weighting

`--features`
Optional feature list used to define or filter the feature space.

`--min-count-per-feature`
Minimum feature total count. Default: `1`.

`--include-feature-regex`, `--exclude-feature-regex`
Regex-based feature filtering.

`--icol-weight`
0-based column index for per-feature weights in `--features`. Default: `-1`, which disables feature weighting. Column 0 of `--features` is the feature name, and column 1 remains the total count when present. Fractional weighted counts are supported.
Negative and non-finite weights are ignored with a warning; zero weights are allowed. Count and regex feature filtering are applied before weights.

`--default-weight`
Default weight for model/prior features missing from `--features` when feature weighting is active. Default: `-1`, which drops missing model/prior features. Set this to a non-negative value, such as `0`, to keep missing model/prior features and fill their weights.

Model fitting and transform use the weighted counts. Pseudobulk output remains on the original count scale.
The topic proportions used by pseudobulk still come from the weighted transform, so weights can affect pseudobulk indirectly through those proportions.
When `--prior-scale-rel` is used with feature weights, the prior is scaled relative to weighted feature totals, so continued fitting or transform should use the same feature weights.

For 10X input, the feature-selection behavior is:

1. If `--features` is provided, it defines the feature space before loading the full matrix.
2. If `--features` is not provided but `--model-prior` is provided, the prior model defines the feature space.
3. If neither is provided and `--min-count-per-feature <= 1`, regex filtering is applied directly to the 10X feature list.
4. Otherwise the 10X matrix is loaded, feature totals are computed, and filtering is applied afterward.

With multiple 10X datasets, the starting 10X feature list in steps 1 and 3 is the intersection across all datasets.

### Transform after fitting

`--transform`
Transform the input units after fitting.

`topic-model --transform`
delegates to `lda-transform`. The shared inference, filtering, weighting,
diagnostic, top-k, and pseudobulk controls are documented once under
[Transform options](#transform-options); their behavior is the same when
invoked through `topic-model`. The delegated transform keeps every non-empty
unit by setting `--min-count 1`.

### LDA fitting options

`--n-topics`
Number of topics.

`--threads`, `--seed`

`--n-epochs`
Number of training passes. Default: `1`.

`--count-cache`
Cache for repeated custom sparse-text training passes:
`off`, `on`, or `auto` (default). `auto` caches when at least two full count
passes are planned. The first pass remains in memory when it fits
`--count-cache-memory-budget`; otherwise it spills to a sequential temporary
file without a random-access index. The 10X path already keeps remapped
documents in memory and does not use this cache.

`--count-cache-memory-budget`
Maximum memory used for retained parsed count data. Integer byte values and
`K`, `M`, or `G` suffixes are accepted. Default: `1G`. A value of `0` forces
sequential temporary storage whenever the count cache is active.

`--temp-dir`
Parent directory for the temporary count cache and any delegated transform
diagnostic spool. The system temporary directory is used when omitted. No
directory is created for a resident cache; temporary files are removed when
their operation finishes.

`--minibatch-size`
Minibatch size. Default: `512`.

`--min-count-train`
Minimum total count per unit for training. Default: `20`.

`--modal`
Modality index for multi-modal custom input.

`--kappa`, `--tau0`
Online SVI learning-rate parameters.

`--alpha`
Document-topic prior. Default: `1/K`.

`--eta`
Topic-word prior. Default: `1/K`.

`--max-iter`
Maximum per-document iterations in inference/transform. Default: `100`.

`--mean-change-tol`
Per-document convergence tolerance. Default: `1e-3`.

`--reproducible-init`
Use deterministic per-document random initialization.

### Prior model and projection options

`--model-prior`
Initialize from an existing model matrix.

`--prior-scale`
Uniform scaling for the prior model.

`--prior-scale-rel`
Scale the prior relative to observed feature totals.

`--projection-only`
Use the prior model to transform the data without fitting. Implies `--transform`.

`--sort-topics`
Sort learned topics by abundance before writing the model.

### Prune topics

`--adaptive-topics`
Treat `--n-topics` as an upper bound. During the final initial-training epoch,
reuse local inference from the most recently processed minibatch-sized window
to estimate topic usage. Then remove topics below the mean-weight threshold,
reset the symmetric LDA priors for the retained topic count, and refit. This
avoids rereading and reinferring the complete training corpus solely for topic
selection.

`--min-topic-mean`
Minimum mean foreground topic weight retained by adaptive fitting. Default:
`1e-5`. This intentionally removes only prior-only slots, not merely rare
programs.

`--adaptive-refit-epochs`
Number of epochs after pruning. Default: `1`.

### Main outputs

`{prefix}.model.tsv`
Feature-by-topic model matrix.

`{prefix}.state.tsv`
SVB state containing the unnormalized global components,
document/topic priors, ordered topic and feature names, and feature-weight
metadata. This is the recommended input to `lda-transform`.

`{prefix}.features.tsv`
Written for 10X fitting when `--features` is not supplied.

`{prefix}.topic_usage.tsv`
Pre-pruning mean foreground weights and retention decisions. Written when
`--adaptive-topics` is used.

Transform outputs are:

`{prefix}.results.tsv`
Topic proportions per unit. By default this contains all topics. With `--topk-only k`, it instead contains columns `K1..Kk` and `P1..Pk`.

`{prefix}.pseudobulk.tsv`
Feature-by-topic pseudobulk counts formed from the transformed topic proportions.

`{prefix}.unit_stats.tsv`
Written when `--residuals` is enabled. Current columns are:

- `total_count`
- `residual`
- `entropy`

With `--unit-diagnostics-similarity`, the file also contains:

- `cosine_sim`
- `sh_lcr`
- `sh_q`

`{prefix}.feature_residuals.tsv`
Written when `--residuals` is enabled. See
[Per-feature diagnostics for topic models](feature_eval.md) for the complete
column schema, formulas, interpretation, LDA specialization, and computational
behavior.

`entropy` is computed from each unit's topic proportions, treating them as a
probability distribution over topics. The opt-in `sh_lcr` and `sh_q`
statistics additionally use cosine similarity among row-normalized topic-word
profiles.

For custom sparse input, carried-over metadata columns from `header_info` appear before the transform outputs. For 10X input, the leading identifier column is `#barcode`. With a single 10X dataset it keeps the original barcode/identifier form; with multiple 10X datasets it is written as `<dataset_id>:<barcode>`.

## `lda-transform`

`lda-transform` applies a fitted plain LDA model to new data. It also accepts
the versioned state written by current `topic-model` fits; the state preserves
the priors and feature weights needed for uncertainty-aware
[partition classification](classifier.md).

### Required

Exactly one of:

- `--in-state`: versioned plain-SVB LDA state (recommended).
- `--in-model`: legacy topic-word model matrix.

`--out-prefix`

Either:

- `--in-data` and `--in-meta`
- or one or more `--in-dge-dir`, or matching repeated `--in-barcodes` + `--in-features` + `--in-matrix`

### Transform options

These options also govern the delegated plain-LDA transform started by
`topic-model --transform`, where applicable.

#### Input processing and inference

`--minibatch-size`
Minibatch size. Default: `512`.

`--modal`
Modality index for multi-modal custom input.

`--threads`, `--seed`, `--verbose`, `--debug`

`--features`, `--min-count-per-feature`, `--include-feature-regex`,
`--exclude-feature-regex`
Define and filter the transform feature space as described under
[Feature selection and weighting](#feature-selection-and-weighting).

`--min-count`
Minimum total count per unit to keep. Default: `20`.

`--icol-weight`, `--default-weight`
Apply feature weights as described under
[Feature selection and weighting](#feature-selection-and-weighting).

`--max-iter`, `--mean-change-tol`

`--sorted-by-barcode`
Use streaming mode for 10X input sorted by barcode. With multiple datasets, streaming follows dataset order first, then barcode order within each dataset.

#### Diagnostic and output controls

`--temp-dir`
Parent directory for temporary diagnostic files. The system temporary
directory is used when omitted, and scoped temporary files are removed when
the transform finishes. A diagnostic directory is created only when
`--residuals` requires transform-data prevalence.

`--residuals`
Write `{prefix}.unit_stats.tsv` and `{prefix}.feature_residuals.tsv`.

`--feature-residuals`
Alias for `--residuals`.

`--feature-diagnostics-cheap`
With `--residuals`, omit the spool-dependent
`adjAbsDiffRate` and `pull` feature columns. This option has no effect when a
fitting command transforms the model's training data, because that path is
already spool-free and always emits raw-residual `pull`.

`--use-training-prevalence`
With `--residuals`, use fitted training prevalence, accumulate raw-residual
Pull inline, and omit `adjAbsDiffRate`. Fitting commands set this automatically
after fitting the transformed data; standalone and projection-only transforms
leave it off by default.

`--unit-diagnostics-similarity`
With `--residuals`, add `cosine_sim`, `sh_lcr`, and `sh_q` to
`{prefix}.unit_stats.tsv`. These statistics require quadratic work in the
number of topics and are disabled by default.

`--topk-only <int>`
Write sparse top-k output to `{prefix}.results.tsv`. The value must be a positive integer.

`--pseudobulk-all-features`
Include every retained input feature in `{prefix}.pseudobulk.tsv`, including features absent from the model. Counts are accumulated on the raw scale as `raw_count * topic_proportion`; extra features do not participate in transformation, residuals, unit totals, or `--min-count`.

`--classifier-model` and `--classifier-*`
Write calibrated fixed-partition predictions and optionally propagate local
LDA posterior uncertainty. See the
[probabilistic partition classifier](classifier.md). A legacy
`--in-model` requires explicit positive `--alpha` unless
`--classifier-plugin-only` is used.

`--out-prefix-classifier`
Use a separate prefix for `.classifications.tsv` and
`.classification_diagnostics.tsv`. It defaults to `--out-prefix`.

`--classifier-only`, `--in-transform-results`
Reuse a previous dense LDA `.results.tsv` as the classifier warm start while
rereading the original counts and fitted state. The source rows must match the
retained input units in order and use the exact dense topic block. This mode
writes only classification output; residual, pseudobulk-all-features, and
top-k factor-output options are not allowed.

### Outputs

`{prefix}.results.tsv`
Per-unit topic proportions, or top-k topic indices/probabilities when `--topk-only` is used.

`{prefix}.pseudobulk.tsv`
Contains model-overlapping features by default, or all retained input features with `--pseudobulk-all-features`. Feature weights are never applied directly to pseudobulk counts.

`{prefix}.unit_stats.tsv`
Written only when `--residuals` is enabled. Its default columns are
`total_count`, `residual`, and `entropy`; `--unit-diagnostics-similarity`
adds `cosine_sim`, `sh_lcr`, and `sh_q`.

`{prefix}.feature_residuals.tsv`
Written only when `--residuals` is enabled. It contains the support,
abundance-shift, deviance, topic-drift, deletion, and Pull diagnostics
defined in [Per-feature diagnostics for topic models](feature_eval.md). Rows
cover model-overlapping features only, including with
`--pseudobulk-all-features`. Most feature diagnostics use effective weighted
counts when feature weights are active. The multinomial variance fields
`F_w`, `Qa_w`, `Q0_w`, `EVES_w`, `TVES_w`, and `U_w` instead use raw integer
counts and raw model-panel totals; feature weights affect them only indirectly
through the inferred topic proportions.

In `{prefix}.unit_stats.tsv`, `total_count` is the raw total count after feature
remapping and filtering but before feature weights are applied.

`{classifier-prefix}.classifications.tsv`
Written when `--classifier-model` is supplied. It contains unit metadata,
prediction diagnostics, LRVB status, and compact or dense class probabilities.
The classifier prefix is `--out-prefix-classifier` when supplied and otherwise
`--out-prefix`.


<!--
### Optional background model

`--fit-background`
Fit a shared background profile in addition to the foreground LDA factors.
Each unit receives a separate `Background` fraction; foreground factor
proportions remain normalized to one independently of that fraction. A warm
start is important because an empirical corpus-average background can
otherwise absorb the whole model before the foreground topics specialize.

`--background-prior`
Input prior for the background profile.

`--background-init-scale`
Scale used when initializing the background from feature totals.

`--background-prevalence-power`
Downweight genes detected in only a small fraction of training units when an
empirical background is constructed. The default `0` constructs a background
proportional only to aggregate counts. Positive powers multiply total
expression by detection prevalence raised to the requested power, then
preserve the original prior mass. They are experimental and can help prevent
abundant cell-type-specific markers from becoming nuisance background.

`--fix-background`
Keep the background profile fixed during training.

`--bg-fraction-prior-a0`, `--bg-fraction-prior-b0`
Beta prior parameters for the background fraction.

`--warm-start-epochs`
Warm-start the topic model before enabling the background. Default: `0.5`.

The fixed-background mode is the recommended starting point. It anchors the
background to aggregate training-set feature frequencies while allowing its
fraction to vary between units. The learned-background mode is experimental.
Background states support dense and top-k transforms and foreground
pseudobulk output. Residual diagnostics and partition-classifier propagation
currently reject background states because those likelihoods have not yet
been extended with the nuisance component. -->

<!-- Additional output from the background model:

`{prefix}.background.tsv`
Background profile and aggregate fitted background fraction. Written when
`--fit-background` is used.

`{prefix}.topic_specificity.tsv`
Foreground topic probabilities and per-gene KL contributions relative to the
background profile. Written when `--fit-background` is used. -->
